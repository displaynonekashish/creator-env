from __future__ import annotations
"""
creator_env_environment.py — Influencer Business Sim v3.0
==========================================================
Full implementation of every mechanic from the design document:

  1.  Multi-turn negotiation (up to 8 turns) with brand_mood state machine
  2.  5 Brand Personalities — HIDDEN from agent, must be inferred from signals
  3.  SHADY brand trap — highest offer is statistically most likely reputation trap
  4.  Multi-niche creator profiles with per-niche reputation scores
  5.  MediaKit system — outdated kit = lowball offers
  6.  Parasocial debt mechanic — must post organic content between deals
  7.  Audience trust — degrades on bad deals, rises on organic and aligned deals
  8.  Fatigue non-linear cliff (0.5 / 0.7 / 0.9 thresholds)
  9.  Exclusivity lockout calendar — opportunity cost mechanic
  10. Gifted deal / early-believer system with viral upside
  11. Payment structure expected value (upfront vs revenue share vs gifted)
  12. Seasonal deal calendar (Q1 freeze → Q4 Black Friday peak)
  13. Brand PR crisis events — kill clause matters
  14. Clause negotiation (exclusivity waiver, revision cap, kill clause)
  15. UNBOX_PRODUCT action — reveal hidden product quality
  16. Integrity test — once per episode, values vs money conflict
  17. Brand relationship memory (partner / blacklisted / early_believer)
  18. Manager NPC unlock at 100K followers
  19. Sparse reward function with exponential sellout penalty
  20. /benchmark endpoint scoring metrics
"""



import random
import math
from typing import Optional, List, Tuple, Dict, Any

from server.models import (
    Observation, BrandDeal, BrandPersonality, BrandMood, NegotiationState,
    Action, DecisionType, MarketSeason, PaymentStructure, MediaKit,
    StepResponse, SEASON_MULTIPLIERS, PAYMENT_CERTAINTY, PAYMENT_MULTIPLIER,
)


# ─── Personality configs ──────────────────────────────────────────────────────

PERSONALITY_CONFIG: Dict[str, Dict] = {
    "luxury": {
        "patience":         2,
        "max_budget_mult":  1.10,
        "rep_risk":         0.05,
        "counter_decay":    0.00,    # barely moves
        # offer = followers * mult / 100  (not /1000 — see _generate_deal)
        "offer_mult_range": (8.0, 18.0),   # 88K followers → $7K–$15.8K
        "crisis_prob":      0.05,
    },
    "startup": {
        "patience":         8,
        "max_budget_mult":  1.25,
        "rep_risk":         0.0,
        "counter_decay":    0.10,
        "offer_mult_range": (1.5, 4.0),    # 88K followers → $1.3K–$3.5K
        "crisis_prob":      0.03,
    },
    "mass_market": {
        "patience":         4,
        "max_budget_mult":  1.50,
        "rep_risk":         0.18,
        "counter_decay":    0.07,
        "offer_mult_range": (3.0, 8.0),    # 88K followers → $2.6K–$7K
        "crisis_prob":      0.10,
    },
    "shady": {
        "patience":         3,
        "max_budget_mult":  2.00,
        "rep_risk":         0.55,
        "counter_decay":    0.03,
        "offer_mult_range": (12.0, 22.0),  # VERY high — the trap signal
        "crisis_prob":      0.40,
    },
    "premium": {
        "patience":         3,
        "max_budget_mult":  1.30,
        "rep_risk":         0.10,
        "counter_decay":    0.06,
        "offer_mult_range": (5.0, 12.0),   # 88K followers → $4.4K–$10.6K
        "crisis_prob":      0.07,
    },
}

# Brand-niche compatibility matrix — crossover bonuses / penalties
NICHE_COMPAT: Dict[str, Dict[str, float]] = {
    "fitness":   {"wellness": 0.9, "food": 0.7, "lifestyle": 0.8, "travel": 0.6},
    "beauty":    {"fashion": 0.9, "lifestyle": 0.8, "wellness": 0.7},
    "tech":      {"education": 0.8, "gaming": 0.7, "finance": 0.6},
    "gaming":    {"tech": 0.8, "education": 0.5},
    "finance":   {"education": 0.8, "tech": 0.6},
    "food":      {"lifestyle": 0.8, "fitness": 0.6, "travel": 0.7},
    "travel":    {"lifestyle": 0.8, "food": 0.7, "fashion": 0.6},
    "fashion":   {"beauty": 0.9, "lifestyle": 0.8},
    "education": {"tech": 0.8, "finance": 0.7},
    "lifestyle": {"fitness": 0.7, "beauty": 0.7, "travel": 0.7, "food": 0.7},
    "wellness":  {"fitness": 0.9, "lifestyle": 0.7},
}

# ─── Deal template pool ───────────────────────────────────────────────────────

DEAL_TEMPLATES = [
    # (brand_id, brand_name,    category,    base_rating, personality,     deliverables, rev_rounds, crisis_p)
    ("nova_tech",    "NovaTech",      "tech",       0.92, "premium",     2, 2, 0.05),
    ("glamour_x",    "GlamourX",      "beauty",     0.88, "luxury",      1, 2, 0.04),
    ("burger_burst", "BurgerBurst",   "food",       0.60, "mass_market", 3, 3, 0.12),
    ("crypto_edge",  "CryptoEdge",    "finance",    0.42, "shady",       1, 1, 0.45),
    ("fit_pulse",    "FitPulse",      "fitness",    0.85, "startup",     2, 2, 0.02),
    ("wander_lux",   "WanderLux",     "travel",     0.80, "luxury",      2, 2, 0.04),
    ("pixel_drop",   "PixelDrop",     "gaming",     0.84, "startup",     1, 1, 0.02),
    ("style_vault",  "StyleVault",    "fashion",    0.68, "mass_market", 4, 4, 0.15),
    ("learn_sphere", "LearnSphere",   "education",  0.91, "startup",     2, 2, 0.01),
    ("vitality_co",  "VitalityCo",    "fitness",    0.52, "shady",       1, 99, 0.50), # unlimited revisions trap
    ("cloud_nest",   "CloudNest",     "tech",       0.82, "premium",     3, 2, 0.06),
    ("urban_eats",   "UrbanEats",     "food",       0.74, "mass_market", 2, 3, 0.10),
    ("quick_cash",   "QuickCash",     "finance",    0.33, "shady",       1, 1, 0.55),
    ("pure_glow",    "PureGlow",      "beauty",     0.95, "luxury",      1, 1, 0.03),
    ("game_forge",   "GameForge",     "gaming",     0.87, "premium",     2, 2, 0.05),
    ("sprout_app",   "SproutApp",     "wellness",   0.79, "startup",     1, 2, 0.01),
    ("meta_threads", "MetaThreads",   "fashion",    0.55, "shady",       3, 99, 0.40),
    ("flow_water",   "FlowWater",     "lifestyle",  0.83, "premium",     1, 2, 0.05),
]

# Integrity test brand — always offered once per episode (task != niche_alignment easy)
INTEGRITY_BRAND = {
    "brand_id":   "fast_feast",
    "brand_name": "FastFeast",
    "category":   "food",
    "base_rating": 0.70,
    "personality": "mass_market",
    "offer_mult":  12.0,  # deliberately tempting — 3x market rate (at new scale)
    "rep_risk":    0.45,
    "description": "Mass-market fast food chain. Offer is extremely lucrative but directly conflicts with wellness/fitness creator values.",
}

# ─── Creator presets ──────────────────────────────────────────────────────────

CREATOR_PRESETS = [
    {
        "niches":          ["tech", "education"],
        "followers":       88_000,
        "engagement_rate": 0.046,
        "values":          ["transparency", "learning"],
    },
    {
        "niches":          ["fitness", "lifestyle", "wellness"],
        "followers":       215_000,
        "engagement_rate": 0.033,
        "values":          ["body_positivity", "no_alcohol", "sustainability"],
    },
    {
        "niches":          ["gaming", "tech"],
        "followers":       155_000,
        "engagement_rate": 0.061,
        "values":          ["authenticity"],
    },
    {
        "niches":          ["beauty", "fashion", "lifestyle"],
        "followers":       330_000,
        "engagement_rate": 0.029,
        "values":          ["sustainability", "cruelty_free"],
    },
    {
        "niches":          ["finance", "education"],
        "followers":       68_000,
        "engagement_rate": 0.057,
        "values":          ["financial_literacy", "transparency"],
    },
]


def _niche_fit(category: str, creator_niches: List[str]) -> float:
    """
    Returns 0.0–1.0 alignment score.
    1.0 = direct match, 0.7–0.9 = compatible crossover, 0.0 = no fit.
    """
    if category in creator_niches:
        return 1.0
    best = 0.0
    for niche in creator_niches:
        compat = NICHE_COMPAT.get(niche, {}).get(category, 0.0)
        best = max(best, compat)
    return best


def _generate_deal(
    followers: int,
    season_mult: float = 1.0,
    force_shady: bool = False,
    force_integrity: bool = False,
    creator_niches: Optional[List[str]] = None,
) -> BrandDeal:
    if force_integrity and creator_niches:
        # Only trigger integrity test for creators with conflicting niches
        ib = INTEGRITY_BRAND
        base_offer = followers * ib["offer_mult"] * season_mult / 300.0
        base_offer = max(500.0, round(base_offer, 2))
        cfg = PERSONALITY_CONFIG[ib["personality"]]
        return BrandDeal(
            brand_id=ib["brand_id"],
            brand_name=ib["brand_name"],
            category=ib["category"],
            brand_rating=ib["base_rating"],
            initial_offer=base_offer,
            current_offer=base_offer,
            personality=BrandPersonality(ib["personality"]),
            patience=cfg["patience"],
            max_budget_mult=cfg["max_budget_mult"],
            rep_risk=ib["rep_risk"],
            payment_structure=PaymentStructure.UPFRONT,
            exclusivity_days=30,
            deliverable_count=2,
            revision_rounds=2,
            has_kill_clause=False,
            is_gifted=False,
            crisis_probability=0.25,
            seasonal_multiplier=season_mult,
            description=ib["description"],
            brief_is_vague=False,
            has_urgency_pressure=False,
            has_verifiable_presence=True,
        )

    tmpl = random.choice(DEAL_TEMPLATES)
    (brand_id, brand_name, category, base_rating,
     personality_str, delivs, revs, crisis_p) = tmpl

    if force_shady:
        personality_str = "shady"

    cfg = PERSONALITY_CONFIG[personality_str]
    low, high = cfg["offer_mult_range"]
    base_offer = followers * random.uniform(low, high) * season_mult / 300.0
    base_offer = max(300.0, round(base_offer, 2))
    rating = min(1.0, max(0.1, base_rating + random.uniform(-0.08, 0.08)))

    # Payment structure — startups sometimes offer revenue share or gifted
    payment = PaymentStructure.UPFRONT
    is_gifted = False
    brand_rev_stage = "established"
    growth_potential = 0.4
    founder_engaged = False
    community_buzz = 0.0

    if personality_str == "startup":
        brand_rev_stage = random.choice(["pre-revenue", "seed", "series_a"])
        growth_potential = random.uniform(0.3, 0.9)
        founder_engaged = random.random() < 0.4
        community_buzz = random.uniform(0.1, 0.6)
        if random.random() < 0.30:
            payment = PaymentStructure.GIFTED_ONLY
            is_gifted = True
        elif random.random() < 0.25:
            payment = PaymentStructure.REVENUE_SHARE
    elif personality_str == "shady":
        payment = PaymentStructure.UPFRONT
        revs = 99  # unlimited revisions — hidden trap

    # Exclusivity
    excl_days = 0
    if personality_str in ("luxury", "premium") and random.random() < 0.5:
        excl_days = random.choice([30, 60, 90])
    elif personality_str == "mass_market" and random.random() < 0.3:
        excl_days = random.choice([14, 30])

    # SHADY observable signals
    brief_vague = (personality_str == "shady") and random.random() < 0.85
    urgency     = (personality_str == "shady") and random.random() < 0.70
    no_presence = (personality_str == "shady") and random.random() < 0.60

    return BrandDeal(
        brand_id=brand_id,
        brand_name=brand_name,
        category=category,
        brand_rating=round(rating, 2),
        initial_offer=base_offer,
        current_offer=base_offer,
        personality=BrandPersonality(personality_str),
        patience=cfg["patience"],
        max_budget_mult=cfg["max_budget_mult"],
        rep_risk=cfg["rep_risk"],
        payment_structure=payment,
        exclusivity_days=excl_days,
        deliverable_count=delivs,
        revision_rounds=revs,
        has_kill_clause=False,
        is_gifted=is_gifted,
        brand_growth_potential=growth_potential,
        founder_engaged=founder_engaged,
        community_buzz=community_buzz,
        brand_revenue_stage=brand_rev_stage,
        brief_is_vague=brief_vague,
        has_urgency_pressure=urgency,
        has_verifiable_presence=not no_presence,
        crisis_probability=crisis_p,
        seasonal_multiplier=season_mult,
        description=f"{personality_str.replace('_',' ').title()} brand in {category}.",
    )


# ─── Environment ─────────────────────────────────────────────────────────────

class CreatorEnvEnvironment:
    """
    OpenEnv-compliant RL environment.

    Episode flow:
      1. reset() → builds creator profile + deal pool
      2. Each step: agent sees Observation with current deal + negotiation state
      3. Agent submits Action (accept/reject/counter/clause/unbox/organic)
      4. Environment resolves action, applies side effects, returns StepResponse
      5. Episode ends at max_steps or creator burnout
      6. Final sparse reward includes exponential sellout penalty
    """

    MAX_STEPS          = 10
    MAX_COUNTER_TURNS  = 8    # design doc: 5–8 turns

    def __init__(self, task: str = "niche_alignment"):
        self.task = task
        self._obs: Optional[Observation] = None
        self._neg: Optional[NegotiationState] = None
        self._current_deal: Optional[BrandDeal] = None
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._step_count: int = 0
        self._deal_pool: List[BrandDeal] = []
        self._deal_history: List[Dict] = []
        self._creator_values: List[str] = []
        self._integrity_test_fired: bool = False

    # ── Public OpenEnv API ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        preset = random.choice(CREATOR_PRESETS)
        self._creator_values = preset.get("values", [])
        season = random.choice(list(MarketSeason))
        s_mult = SEASON_MULTIPLIERS[season.value]

        niches = preset["niches"]
        followers = preset["followers"]
        eng = preset["engagement_rate"]

        # Build MediaKit
        kit = MediaKit(
            followers=followers,
            engagement_rate=eng,
            primary_niche=niches[0],
            secondary_niches=niches[1:],
            content_quality_score=round(random.uniform(0.70, 0.92), 2),
            is_competitive=(eng > 0.03),
        )

        # Per-niche reputation starts at different levels
        niche_rep = {n: round(random.uniform(0.65, 0.92), 2) for n in niches}

        self._obs = Observation(
            creator_niches=niches,
            niche_reputation=niche_rep,
            followers=followers,
            engagement_rate=eng,
            media_kit=kit,
            reputation=round(sum(niche_rep.values()) / len(niche_rep), 2),
            audience_trust=0.82,
            fatigue=0.0,
            parasocial_debt=0,
            consecutive_bad_deals=0,
            manager_available=(followers >= 100_000),
            earnings=0.0,
            monthly_target=self._monthly_target(),
            market_season=season,
            seasonal_multiplier=s_mult,
            lockout_calendar={},
            deals_accepted=0,
            deals_rejected=0,
            sellout_count=0,
            shady_deals_detected=0,
            episode_step=0,
            max_steps=self.MAX_STEPS,
            brand_relationships={},
            early_believer_brands=[],
        )

        self._cumulative_reward = 0.0
        self._done = False
        self._step_count = 0
        self._deal_history = []
        self._integrity_test_fired = False

        self._deal_pool = self._build_pool(followers, s_mult, niches)
        self._present_next_deal()
        return self._obs

    def step(self, action: Action) -> StepResponse:
        if self._done:
            return StepResponse(
                observation=self._obs,
                reward=0.0, done=True,
                metadata={"error": "episode already done"},
            )

        self._step_count += 1
        self._obs.episode_step = self._step_count
        self._tick_lockouts()

        reward, info = self._process_action(action)
        self._cumulative_reward += reward

        # Burnout check
        if self._obs.fatigue >= 1.0:
            self._done = True
            info["burnout"] = True
            info["final_score"] = 0.0
            return StepResponse(observation=self._obs, reward=reward, done=True, metadata=info)

        # Trust crisis check
        if self._obs.consecutive_bad_deals >= 3:
            self._obs.audience_trust = max(0.0, self._obs.audience_trust - 0.20)
            self._obs.followers = int(self._obs.followers * 0.92)
            self._obs.consecutive_bad_deals = 0
            info["trust_crisis"] = True

        if self._step_count >= self.MAX_STEPS:
            self._done = True

        if self._done:
            info["final_score"] = self._calculate_final_score()

        return StepResponse(
            observation=self._obs,
            reward=round(reward, 4),
            done=self._done,
            metadata=info,
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "observation": self._obs.model_dump() if self._obs else {},
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
        }

    # ── Action router ─────────────────────────────────────────────────────────

    def _process_action(self, action: Action) -> Tuple[float, Dict]:
        info: Dict[str, Any] = {}
        deal = self._current_deal
        neg  = self._neg

        # ── No deal ───────────────────────────────────────────────────────────
        if deal is None:
            self._present_next_deal()
            return 0.0, {"note": "no active deal, new deal presented"}

        d = action.decision

        # ── Organic content post ──────────────────────────────────────────────
        if d == DecisionType.POST_ORGANIC_CONTENT:
            return self._action_organic(info)

        # ── Unbox product ─────────────────────────────────────────────────────
        if d == DecisionType.UNBOX_PRODUCT:
            return self._action_unbox(deal, info)

        # ── Clause requests ───────────────────────────────────────────────────
        if d in (DecisionType.REQUEST_KILL_CLAUSE,
                 DecisionType.REQUEST_EXCLUSIVITY_WAIVER,
                 DecisionType.REQUEST_REVISION_CAP):
            return self._action_clause(d, deal, neg, info)

        # ── Accept ────────────────────────────────────────────────────────────
        if d == DecisionType.ACCEPT:
            return self._action_accept(deal, info)

        # ── Reject ────────────────────────────────────────────────────────────
        if d == DecisionType.REJECT:
            return self._action_reject(deal, info)

        # ── Counter ───────────────────────────────────────────────────────────
        if d in (DecisionType.COUNTER_10, DecisionType.COUNTER_20,
                 DecisionType.COUNTER_40, DecisionType.COUNTER_60):
            return self._action_counter(d, action.counter_price, deal, neg, info)

        return 0.0, {"error": f"unrecognised action: {d}"}

    # ── Accept ────────────────────────────────────────────────────────────────

    def _action_accept(self, deal: BrandDeal, info: Dict) -> Tuple[float, Dict]:
        obs = self._obs
        aligned_score = _niche_fit(deal.category, obs.creator_niches)
        is_sellout = deal.rep_risk > 0.4 or aligned_score < 0.4

        # Integrity test check
        if obs.integrity_test_triggered and obs.integrity_test_passed is None:
            conflict = any(
                v in ("body_positivity", "no_alcohol", "sustainability", "wellness")
                for v in self._creator_values
            ) and deal.category == "food" and deal.rep_risk > 0.3
            if conflict:
                obs.integrity_test_passed = False
                obs.integrity_test_triggered = False   # FIX: clear flag after resolution
                obs.audience_trust = max(0.0, obs.audience_trust - 0.35)
                for niche in obs.creator_niches:
                    if niche in ("fitness", "wellness", "lifestyle"):
                        obs.niche_reputation[niche] = max(
                            0.0, obs.niche_reputation.get(niche, 0.8) - 0.40
                        )
                info["integrity_failed"] = True
                info["note"] = "Accepted integrity-test brand — permanent trust damage"
            else:
                # Not a conflicting deal — clear the flag, no penalty
                obs.integrity_test_triggered = False

        # Earnings — expected value based on payment structure
        cert = PAYMENT_CERTAINTY.get(deal.payment_structure.value, 1.0)
        mult = PAYMENT_MULTIPLIER.get(deal.payment_structure.value, 1.0)
        cash = deal.current_offer * cert * mult
        obs.earnings += cash

        # Reputation effects
        if is_sellout:
            rep_hit = deal.rep_risk + (0.12 if aligned_score < 0.4 else 0.0)
            obs.reputation = max(0.0, obs.reputation - rep_hit)
            obs.audience_trust = max(0.0, obs.audience_trust - deal.rep_risk * 0.5)
            obs.sellout_count += 1
            obs.consecutive_bad_deals += 1
            info["reputation_hit"] = round(rep_hit, 3)
            # Update niche rep
            for niche in obs.creator_niches:
                if niche != deal.category:
                    obs.niche_reputation[niche] = max(
                        0.0, obs.niche_reputation.get(niche, 0.8) - 0.05
                    )
        else:
            rep_gain = 0.02 + (aligned_score - 0.5) * 0.04
            obs.reputation = min(1.0, obs.reputation + rep_gain)
            obs.audience_trust = min(1.0, obs.audience_trust + 0.02)
            obs.consecutive_bad_deals = 0
            obs.niche_reputation[deal.category] = min(
                1.0, obs.niche_reputation.get(deal.category, 0.8) + 0.03
            )

        # Gifted early-believer upside
        if deal.is_gifted and deal.brand_revenue_stage in ("pre-revenue", "seed"):
            obs.early_believer_brands.append(deal.brand_id)
            obs.brand_relationships[deal.brand_id] = "early_believer"
            # Viral chance
            viral_p = deal.brand_growth_potential * 0.3 + (
                0.4 if deal.payment_structure == PaymentStructure.GIFTED_ONLY else 0.0
            )
            if random.random() < viral_p:
                obs.followers = int(obs.followers * 1.15)
                obs.audience_trust = min(1.0, obs.audience_trust + 0.12)
                obs.reputation = min(1.0, obs.reputation + 0.08)
                info["viral_moment"] = True
                info["early_believer_viral"] = deal.brand_name
        else:
            # Update relationship
            existing = obs.brand_relationships.get(deal.brand_id, "prospect")
            obs.brand_relationships[deal.brand_id] = (
                "partner" if existing == "warm" else "warm"
            )

        # Exclusivity lockout
        if deal.exclusivity_days > 0:
            obs.lockout_calendar[deal.category] = (
                obs.lockout_calendar.get(deal.category, 0) + deal.exclusivity_days
            )
            info["locked_out_category"] = deal.category
            info["lockout_days"] = deal.exclusivity_days

        # Parasocial debt
        obs.parasocial_debt += 1
        if obs.parasocial_debt > 3:
            obs.engagement_rate = max(0.005, obs.engagement_rate * 0.88)
            info["parasocial_debt_warning"] = obs.parasocial_debt

        # PR crisis post-deal
        if random.random() < deal.crisis_probability:
            if deal.has_kill_clause or "kill_clause" in (self._neg.clauses_granted if self._neg else []):
                obs.reputation = min(1.0, obs.reputation + 0.05)
                info["pr_crisis_handled"] = True
            else:
                obs.reputation = max(0.0, obs.reputation - 0.25)
                obs.followers  = int(obs.followers * 0.92)
                info["pr_crisis"] = True
                info["kill_clause_missing_cost"] = True

        # Fatigue — true cost
        self._apply_fatigue(deal.true_fatigue_cost)

        # Follower growth
        self._grow_followers(deal, aligned_score)

        obs.deals_accepted += 1
        self._deal_history.append({
            "brand_id": deal.brand_id, "value": cash,
            "aligned": aligned_score > 0.5, "sellout": is_sellout,
        })

        reward = self._reward_accept(deal, aligned_score, cash)
        info["outcome"]    = "accepted"
        info["deal_value"] = round(cash, 2)
        self._present_next_deal()
        return reward, info

    # ── Reject ────────────────────────────────────────────────────────────────

    def _action_reject(self, deal: BrandDeal, info: Dict) -> Tuple[float, Dict]:
        obs = self._obs
        aligned_score = _niche_fit(deal.category, obs.creator_niches)
        is_shady       = deal.personality == BrandPersonality.SHADY
        is_integrity   = obs.integrity_test_triggered and obs.integrity_test_passed is None

        # Track correct SHADY rejection
        if is_shady:
            obs.shady_deals_detected += 1
            obs.reputation = min(1.0, obs.reputation + 0.03)
            obs.audience_trust = min(1.0, obs.audience_trust + 0.04)
            info["shady_detected_correctly"] = True

        # Integrity test pass — CLEAR the flag so future deals are not affected
        if is_integrity:
            obs.integrity_test_passed = True
            obs.integrity_test_triggered = False   # FIX: clear flag after resolution
            obs.reputation = min(1.0, obs.reputation + 0.15)
            obs.audience_trust = min(1.0, obs.audience_trust + 0.08)
            info["integrity_passed"] = True
            info["note"] = "Rejected integrity-test brand — reputation premium earned"

        # Update relationship — ghosting vs graceful rejection
        existing = obs.brand_relationships.get(deal.brand_id, "prospect")
        obs.brand_relationships[deal.brand_id] = (
            "blacklisted" if existing == "warm" else "prospect"
        )

        self._apply_fatigue(0.01)
        obs.deals_rejected += 1
        reward = self._reward_reject(deal, aligned_score, is_shady, is_integrity)
        info["outcome"] = "rejected"
        self._present_next_deal()
        return reward, info

    # ── Counter ───────────────────────────────────────────────────────────────

    def _action_counter(
        self,
        d: DecisionType,
        custom_price: Optional[float],
        deal: BrandDeal,
        neg: NegotiationState,
        info: Dict,
    ) -> Tuple[float, Dict]:
        obs = self._obs

        # Determine multiplier from action type
        pct_map = {
            DecisionType.COUNTER_10: 1.10,
            DecisionType.COUNTER_20: 1.20,
            DecisionType.COUNTER_40: 1.40,
            DecisionType.COUNTER_60: 1.60,
        }
        mult = pct_map.get(d, 1.20)
        proposed = custom_price if custom_price and custom_price > deal.current_offer else round(deal.current_offer * mult, 2)

        # Anchoring bonus — first counter gets a small edge
        if neg.turn == 0:
            neg.agent_anchored = True

        neg.turn += 1
        neg.last_counter_by_creator = proposed
        self._apply_fatigue(0.04)

        cfg = PERSONALITY_CONFIG[deal.personality.value]

        # ── Brand mood update ──────────────────────────────────────────────────
        if mult >= 1.60:
            neg.brand_mood = BrandMood.ANGRY
            info["brand_reaction"] = "Brand is angry at the aggressive ask"
        elif neg.brand_patience_remaining == 1:
            neg.brand_mood = BrandMood.DESPERATE
            info["brand_reaction"] = "Brand is desperate — last chance to close"
        elif neg.brand_patience_remaining <= 2:
            neg.brand_mood = BrandMood.COOLING
        else:
            neg.brand_mood = BrandMood.INTERESTED

        # ── Patience exhausted ────────────────────────────────────────────────
        if neg.brand_patience_remaining <= 0:
            neg.brand_walked_away = True
            obs.deals_rejected += 1
            obs.brand_relationships[deal.brand_id] = "blacklisted"
            info["outcome"] = "brand_walked_away"
            info["note"] = f"{deal.personality.value} brand ran out of patience after {neg.turn} turns"
            self._present_next_deal()
            return -0.05, info

        # ── Brand responds ────────────────────────────────────────────────────
        neg.brand_patience_remaining -= 1
        decay = cfg["counter_decay"] * neg.turn
        # Angry brand refuses to raise
        if neg.brand_mood == BrandMood.ANGRY:
            decay = 1.0
        max_offer = deal.initial_offer * deal.max_budget_mult
        brand_raise_frac = max(0.0, 0.55 - decay)
        brand_counter = deal.current_offer + (max_offer - deal.current_offer) * brand_raise_frac
        brand_counter = round(min(max_offer, brand_counter), 2)
        neg.last_counter_by_brand = brand_counter

        # ── Deal closes if brand meets creator ask ────────────────────────────
        if brand_counter >= proposed * 0.93:
            deal.current_offer = brand_counter
            # Close deal
            self._current_deal = deal
            return self._action_accept(deal, {
                "outcome": "negotiated_accepted",
                "negotiation_turns": neg.turn,
                "anchoring_bonus": neg.agent_anchored,
            })

        # Brand makes partial counter — stays alive
        deal.current_offer = brand_counter
        obs.deal = deal
        obs.negotiation = neg

        info["outcome"]          = "brand_countered"
        info["brand_new_offer"]  = brand_counter
        info["patience_left"]    = neg.brand_patience_remaining
        info["brand_mood"]       = neg.brand_mood.value
        return 0.01, info

    # ── Clause requests ───────────────────────────────────────────────────────

    def _action_clause(
        self, d: DecisionType, deal: BrandDeal, neg: Optional[NegotiationState], info: Dict
    ) -> Tuple[float, Dict]:
        obs = self._obs
        cfg = PERSONALITY_CONFIG[deal.personality.value]

        clause_map = {
            DecisionType.REQUEST_KILL_CLAUSE:        "kill_clause",
            DecisionType.REQUEST_EXCLUSIVITY_WAIVER: "exclusivity_waiver",
            DecisionType.REQUEST_REVISION_CAP:       "revision_cap",
        }
        clause = clause_map[d]

        # Luxury / SHADY brands have low tolerance for clause requests
        grant_prob = {
            "luxury":      0.3,
            "startup":     0.8,
            "mass_market": 0.6,
            "shady":       0.1,   # SHADY almost always refuses — signal
            "premium":     0.65,
        }.get(deal.personality.value, 0.5)

        if random.random() < grant_prob:
            if neg:
                neg.clauses_granted.append(clause)
            if clause == "kill_clause":
                deal.has_kill_clause = True
            elif clause == "exclusivity_waiver":
                deal.exclusivity_days = 0
            elif clause == "revision_cap":
                deal.revision_rounds = min(deal.revision_rounds, 2)
            info["clause_granted"]  = clause
            info["outcome"]         = "clause_granted"
            self._apply_fatigue(0.02)
            obs.deal = deal
            obs.negotiation = neg
            return 0.03, info
        else:
            info["clause_denied"] = clause
            info["outcome"]       = "clause_denied"
            if neg:
                neg.brand_patience_remaining -= 1
            self._apply_fatigue(0.01)
            obs.deal = deal
            obs.negotiation = neg
            return 0.0, info

    # ── Unbox product ─────────────────────────────────────────────────────────

    def _action_unbox(self, deal: BrandDeal, info: Dict) -> Tuple[float, Dict]:
        """Reveal hidden product_quality — useful for evaluating gifted/startup deals."""
        if deal.product_quality is None:
            deal.product_quality = round(deal.brand_growth_potential * random.uniform(0.7, 1.1), 2)
            deal.product_quality = min(1.0, deal.product_quality)
        self._obs.deal = deal
        self._apply_fatigue(0.01)
        info["outcome"]          = "product_unboxed"
        info["product_quality"]  = deal.product_quality
        info["note"] = (
            "High quality product — likely genuine." if deal.product_quality > 0.7
            else "Average product." if deal.product_quality > 0.4
            else "Low quality — suspicious for the price offered."
        )
        return 0.0, info

    # ── Organic content ───────────────────────────────────────────────────────

    def _action_organic(self, info: Dict) -> Tuple[float, Dict]:
        obs = self._obs
        obs.parasocial_debt = max(0, obs.parasocial_debt - 2)
        obs.audience_trust  = min(1.0, obs.audience_trust + 0.03)
        obs.engagement_rate = min(1.0, obs.engagement_rate * 1.05)
        self._apply_fatigue(0.03)   # costs time, no money
        info["outcome"] = "organic_posted"
        info["parasocial_debt"] = obs.parasocial_debt
        return 0.02, info   # small reward for maintaining audience health

    # ── Reward functions ──────────────────────────────────────────────────────

    def _reward_accept(self, deal: BrandDeal, aligned_score: float, cash: float) -> float:
        obs = self._obs

        # Revenue progress
        rev = min(1.0, obs.earnings / obs.monthly_target) * 0.40

        # Alignment bonus / penalty
        if aligned_score >= 0.9:
            align = 0.18
        elif aligned_score >= 0.6:
            align = 0.08
        elif aligned_score >= 0.3:
            align = -0.05
        else:
            align = -0.22

        # Trust health
        trust = obs.audience_trust * 0.15

        # Fatigue penalty
        fat_pen = obs.fatigue * 0.10

        # Sellout penalty — exponential decay: 0.3^i per sellout
        sellout_pen = sum(
            0.3 ** i for i in range(obs.sellout_count)
        ) * 0.15

        # Engagement health
        eng = min(obs.engagement_rate * 5, 0.10)

        raw = rev + align + trust - fat_pen - sellout_pen + eng
        return round(max(-0.5, min(1.0, raw)), 4)

    def _reward_reject(
        self, deal: BrandDeal, aligned_score: float, is_shady: bool, is_integrity: bool
    ) -> float:
        if is_shady:
            return 0.18   # correctly identified and avoided the trap
        if is_integrity:
            return 0.20   # values reasoning — highest reject reward
        if deal.rep_risk > 0.4:
            return 0.12   # correctly avoided rep damage
        if aligned_score < 0.3:
            return 0.08   # good call — not your niche
        if aligned_score >= 0.7 and deal.brand_rating >= 0.75:
            return -0.06  # opportunity cost — probably should have taken it
        return 0.02

    # ── Side effects ──────────────────────────────────────────────────────────

    def _apply_fatigue(self, delta: float):
        obs = self._obs
        obs.fatigue = round(min(1.0, obs.fatigue + delta), 4)
        # Non-linear cliff thresholds
        if obs.fatigue >= 0.90:
            obs.engagement_rate = max(0.002, obs.engagement_rate * 0.88)
        elif obs.fatigue >= 0.70:
            obs.engagement_rate = max(0.005, obs.engagement_rate * 0.95)

    def _grow_followers(self, deal: BrandDeal, aligned_score: float):
        obs = self._obs
        base = 0.04 * aligned_score - 0.02 * (1 - aligned_score)
        rep_mult    = obs.reputation
        trust_mult  = obs.audience_trust
        sellout_adj = -0.01 * obs.sellout_count
        rate = (base + sellout_adj) * rep_mult * trust_mult
        delta = int(obs.followers * rate)
        obs.followers = max(1000, obs.followers + delta)
        # Manager unlock at 100K
        if obs.followers >= 100_000 and not obs.manager_available:
            obs.manager_available = True

    def _tick_lockouts(self):
        """Decay exclusivity lockout windows each step."""
        to_del = []
        for cat, days in self._obs.lockout_calendar.items():
            remaining = days - 1
            if remaining <= 0:
                to_del.append(cat)
            else:
                self._obs.lockout_calendar[cat] = remaining
        for cat in to_del:
            del self._obs.lockout_calendar[cat]

    # ── Deal pool builder ─────────────────────────────────────────────────────

    def _build_pool(
        self, followers: int, season_mult: float, creator_niches: List[str]
    ) -> List[BrandDeal]:
        pool = []
        # Guarantee one SHADY deal in the pool (the trap)
        pool.append(_generate_deal(followers, season_mult, force_shady=True))
        # Guarantee one integrity test for rep-sensitive tasks
        if self.task in ("reputation_protection", "target_negotiation"):
            integrity_niches_present = any(
                n in creator_niches for n in ("fitness", "wellness", "lifestyle")
            )
            if integrity_niches_present:
                pool.append(_generate_deal(
                    followers, season_mult,
                    force_integrity=True, creator_niches=creator_niches
                ))
        # Fill rest
        while len(pool) < self.MAX_STEPS + 3:
            pool.append(_generate_deal(followers, season_mult))
        random.shuffle(pool)
        return pool

    def _present_next_deal(self):
        obs = self._obs
        if not self._deal_pool:
            obs.deal = None
            obs.negotiation = None
            self._done = True
            return

        # Skip locked-out categories
        attempts = 0
        while self._deal_pool and attempts < len(self._deal_pool):
            candidate = self._deal_pool[0]
            if candidate.category in obs.lockout_calendar and obs.lockout_calendar[candidate.category] > 0:
                self._deal_pool.append(self._deal_pool.pop(0))
                attempts += 1
                continue
            break

        if not self._deal_pool:
            obs.deal = None
            obs.negotiation = None
            self._done = True
            return

        deal = self._deal_pool.pop(0)

        # Integrity test trigger
        if (
            not self._integrity_test_fired
            and deal.brand_id == "fast_feast"
            and self.task in ("reputation_protection", "target_negotiation")
        ):
            self._integrity_test_fired = True
            obs.integrity_test_triggered = True
            obs.integrity_test_passed   = None

        self._current_deal = deal
        self._neg = NegotiationState(
            brand_patience_remaining=deal.patience,
        )
        obs.deal        = deal
        obs.negotiation = self._neg

    # ── Final score (sparse, long-term) ──────────────────────────────────────

    def _calculate_final_score(self) -> float:
        """
        Season sustainability score — geometric weighting ensures one bad
        dimension can't be hidden by excellence elsewhere.

        Components:
          followers × audience_trust   (30%) — followers only count if trusted
          revenue vs target            (30%) — hit the target
          reputation                  (20%) — long-term brand health
          1 - fatigue                  (15%) — sustainability
          brand_diversity              (5%)  — avoid single-brand dependency

        Penalties (applied after):
          - Exponential sellout penalty
          - Trust crisis penalty
          - Integrity test failure penalty
        """
        obs = self._obs
        hist = self._deal_history

        # Follower value — zero if trust < 0.6 (audience doesn't believe creator)
        trust_factor = obs.audience_trust if obs.audience_trust >= 0.60 else 0.0
        follower_val = min(1.0, (obs.followers / 500_000) * trust_factor)

        rev = min(1.0, obs.earnings / obs.monthly_target)
        rep = obs.reputation
        sust = max(0.0, 1.0 - obs.fatigue ** 2)  # non-linear — burnout is catastrophic

        brand_div = 0.5
        if hist:
            unique_brands = len(set(d["brand_id"] for d in hist))
            brand_div = min(1.0, unique_brands / max(1, len(hist)))

        if self.task == "niche_alignment":
            base = follower_val*0.30 + rev*0.25 + rep*0.30 + sust*0.10 + brand_div*0.05
        elif self.task == "reputation_protection":
            base = follower_val*0.20 + rev*0.20 + rep*0.45 + sust*0.10 + brand_div*0.05
        else:  # target_negotiation
            base = follower_val*0.20 + rev*0.45 + rep*0.20 + sust*0.10 + brand_div*0.05

        # Exponential sellout penalty — 0.3^i per sellout
        sellout_pen = sum(0.3 ** i for i in range(obs.sellout_count)) * 0.12

        # Integrity test failure
        int_pen = 0.15 if obs.integrity_test_passed is False else 0.0

        # Shady brand accepted = hidden penalty (wasn't counted as sellout above)
        # (already captured in rep_risk → sellout_count)

        final = max(0.0, min(1.0, base - sellout_pen - int_pen))
        return round(final, 4)

    # ── Monthly target ────────────────────────────────────────────────────────

    def _monthly_target(self) -> float:
        return {
            "niche_alignment":       4000.0,
            "reputation_protection": 3000.0,
            "target_negotiation":    6250.0,
        }.get(self.task, 5000.0)