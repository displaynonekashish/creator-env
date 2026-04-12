"""
models.py — Influencer Business Sim v3.0
=========================================
Complete data structures implementing:
  - Multi-niche creators with per-niche reputation scores
  - 5 Brand Personalities (LUXURY, STARTUP, MASS_MARKET, SHADY, PREMIUM)
    — agent is NEVER told personality, must infer from signals
  - Full NegotiationState with brand_mood & anchoring bonus
  - MediaKit (creator's market-facing profile, affects offer quality)
  - 14-dimension Observation space
  - Parasocial debt, audience trust, lockout calendar
  - Seasonal market state with multipliers
  - Gifted deal / early-believer system
  - Extended action space (counter tiers, clause requests, organic post)
  - Payment structure types with expected value logic
  - Brand relationship memory (rolodex)
  - Integrity test tracking
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class DecisionType(str, Enum):
    ACCEPT                     = "accept"
    REJECT                     = "reject"
    COUNTER_10                 = "counter_10"          # +10% above current offer
    COUNTER_20                 = "counter_20"          # +20%
    COUNTER_40                 = "counter_40"          # +40%
    COUNTER_60                 = "counter_60"          # +60% — risky with luxury/shady
    REQUEST_EXCLUSIVITY_WAIVER = "request_exclusivity_waiver"
    REQUEST_REVISION_CAP       = "request_revision_cap"
    REQUEST_KILL_CLAUSE        = "request_kill_clause"
    UNBOX_PRODUCT              = "unbox_product"       # reveal hidden product_quality
    POST_ORGANIC_CONTENT       = "post_organic_content"  # reduce parasocial_debt


class BrandPersonality(str, Enum):
    """
    HIDDEN from agent — must be inferred from observable deal signals.
    LUXURY      patience=2, max_budget_mult=1.10, high initial offer, hates <500K followers
    STARTUP     patience=8, max_budget_mult=1.25, low offer, equity bonus for early accept
    MASS_MARKET patience=4, max_budget_mult=1.50, wants virality, penalises niche audiences
    SHADY       patience=3, offer 40%+ above market, vague brief, -30 rep if accepted
    PREMIUM     patience=3, max_budget_mult=1.30, balanced and reasonable
    """
    LUXURY      = "luxury"
    STARTUP     = "startup"
    MASS_MARKET = "mass_market"
    SHADY       = "shady"
    PREMIUM     = "premium"


class BrandMood(str, Enum):
    INTERESTED = "interested"
    COOLING    = "cooling"    # patience dropping
    DESPERATE  = "desperate"  # patience == 1 → push harder now
    ANGRY      = "angry"      # agent asked for 60%+ above offer
    WALKED     = "walked"


class PaymentStructure(str, Enum):
    UPFRONT       = "upfront"        # certainty=1.0,  cash_mult=1.0
    MILESTONE     = "milestone"      # certainty=0.85, cash_mult=1.15
    REVENUE_SHARE = "revenue_share"  # certainty=0.3,  cash_mult=2.5 — high variance
    GIFTED_ONLY   = "gifted_only"    # certainty=1.0,  cash=0


class MarketSeason(str, Enum):
    Q1_JAN = "q1_jan"   # multiplier 0.60 — post-holiday freeze
    Q1_FEB = "q1_feb"   # multiplier 0.80 — Valentine's push
    Q2     = "q2"       # multiplier 0.90 — steady
    Q3_JUL = "q3_jul"   # multiplier 0.75 — summer slump
    Q4_OCT = "q4_oct"   # multiplier 1.30 — holiday ramp-up
    Q4_NOV = "q4_nov"   # multiplier 1.50 — BLACK FRIDAY peak
    Q4_DEC = "q4_dec"   # multiplier 1.40 — Christmas push


SEASON_MULTIPLIERS: Dict[str, float] = {
    "q1_jan": 0.60, "q1_feb": 0.80, "q2": 0.90,
    "q3_jul": 0.75, "q4_oct": 1.30, "q4_nov": 1.50, "q4_dec": 1.40,
}

PAYMENT_CERTAINTY: Dict[str, float] = {
    "upfront": 1.0, "milestone": 0.85, "revenue_share": 0.30, "gifted_only": 1.0,
}
PAYMENT_MULTIPLIER: Dict[str, float] = {
    "upfront": 1.0, "milestone": 1.15, "revenue_share": 2.50, "gifted_only": 0.0,
}


# ─── Media Kit ────────────────────────────────────────────────────────────────

class MediaKit(BaseModel):
    """
    Creator's market-facing profile. Brands check this before making an offer.
    Outdated or weak kit → lowball offers. Strong kit → premium offers.
    Past prestigious partners signal legitimacy to incoming brands.
    """
    followers:              int
    engagement_rate:        float
    primary_niche:          str
    secondary_niches:       List[str]       = []
    audience_age_18_24_pct: float           = 0.55
    audience_geo_us_pct:    float           = 0.70
    past_brand_partners:    List[str]       = []   # brands judge by the company you keep
    avg_deal_value:         float           = 0.0
    content_quality_score:  float           = Field(0.75, ge=0.0, le=1.0)
    is_competitive:         bool            = True  # eng>0.03 & quality>0.70


# ─── Brand / Deal ─────────────────────────────────────────────────────────────

class BrandDeal(BaseModel):
    brand_id:               str
    brand_name:             str
    category:               str                     # should match a CreatorNiche value
    brand_rating:           float  = Field(..., ge=0.0, le=1.0)
    initial_offer:          float  = Field(..., gt=0)
    current_offer:          float  = Field(..., gt=0)
    personality:            BrandPersonality        # HIDDEN — not shown to agent in prompt

    # Negotiation config (derived from personality)
    patience:               int    = Field(..., ge=0)
    max_budget_mult:        float  = Field(..., ge=1.0)
    rep_risk:               float  = Field(0.0, ge=0.0, le=1.0)

    # Contract terms
    payment_structure:      PaymentStructure = PaymentStructure.UPFRONT
    exclusivity_days:       int    = 0
    deliverable_count:      int    = 1
    revision_rounds:        int    = 2      # 99 = "unlimited" (hidden trap)
    has_kill_clause:        bool   = False

    # Gifted / early-believer system
    is_gifted:              bool   = False
    brand_growth_potential: float  = Field(0.5, ge=0.0, le=1.0)  # HIDDEN
    founder_engaged:        bool   = False            # VISIBLE signal
    community_buzz:         float  = Field(0.0, ge=0.0, le=1.0)  # VISIBLE signal
    product_quality:        Optional[float] = None    # revealed only via UNBOX_PRODUCT
    brand_revenue_stage:    str    = "established"    # "pre-revenue"|"seed"|"series_a"|"established"

    # SHADY brand observable signals (agent must learn to read)
    brief_is_vague:             bool = False  # VISIBLE — "just be authentic!"
    has_urgency_pressure:       bool = False  # VISIBLE — "offer expires in 1 hour"
    has_verifiable_presence:    bool = True   # VISIBLE — False means no website found

    # Post-deal risk
    crisis_probability:     float  = Field(0.0, ge=0.0, le=1.0)
    seasonal_multiplier:    float  = 1.0
    description:            str    = ""

    @property
    def true_fatigue_cost(self) -> float:
        """True cost including deliverables + revision rounds."""
        revs = min(self.revision_rounds, 5)
        return round(0.04 + (self.deliverable_count * 0.015) + (revs * 0.008), 3)

    @property
    def expected_cash_value(self) -> float:
        """Expected value = offer × certainty × payment_multiplier."""
        cert = PAYMENT_CERTAINTY.get(self.payment_structure.value, 1.0)
        mult = PAYMENT_MULTIPLIER.get(self.payment_structure.value, 1.0)
        return round(self.current_offer * cert * mult, 2)


# ─── Negotiation State ────────────────────────────────────────────────────────

class NegotiationState(BaseModel):
    """Live state of the current multi-turn negotiation (up to 8 turns)."""
    turn:                     int       = 0
    brand_patience_remaining: int       = 3
    brand_mood:               BrandMood = BrandMood.INTERESTED
    last_counter_by_creator:  Optional[float] = None
    last_counter_by_brand:    Optional[float] = None
    agent_anchored:           bool      = False   # anchoring bonus if agent counters first
    brand_walked_away:        bool      = False
    deal_closed:              bool      = False
    clauses_granted:          List[str] = []      # e.g. "kill_clause", "revision_cap"


# ─── Full Observation (14 dimensions) ────────────────────────────────────────

class Observation(BaseModel):
    # 1. Creator profile
    creator_niches:           List[str]
    niche_reputation:         Dict[str, float]  = {}   # per-niche score 0–1
    followers:                int
    engagement_rate:          float             = Field(..., ge=0.0, le=1.0)
    media_kit:                Optional[MediaKit] = None

    # 2. Creator health
    reputation:               float             = Field(..., ge=0.0, le=1.0)
    audience_trust:           float             = Field(0.80, ge=0.0, le=1.0)
    fatigue:                  float             = Field(0.0,  ge=0.0, le=1.0)
    parasocial_debt:          int               = 0   # >3 → engagement drop
    consecutive_bad_deals:    int               = 0   # ≥3 → trust crisis
    manager_available:        bool              = False
    manager_signed:           bool              = False

    # 3. Financials
    earnings:                 float             = 0.0
    monthly_target:           float             = 5000.0

    # 4. Market state
    market_season:            MarketSeason      = MarketSeason.Q2
    seasonal_multiplier:      float             = 1.0
    platform_format:          str               = "video"   # can shift mid-season

    # 5. Lockout calendar
    lockout_calendar:         Dict[str, int]    = {}   # category → days remaining

    # 6. Current negotiation
    deal:                     Optional[BrandDeal]         = None
    negotiation:              Optional[NegotiationState]  = None

    # 7. Episode counters
    deals_accepted:           int               = 0
    deals_rejected:           int               = 0
    sellout_count:            int               = 0
    shady_deals_detected:     int               = 0
    integrity_test_triggered: bool              = False
    integrity_test_passed:    Optional[bool]    = None
    episode_step:             int               = 0
    max_steps:                int               = 10

    # 8. Relationship memory
    brand_relationships:      Dict[str, str]    = {}   # brand_id → status
    early_believer_brands:    List[str]         = []


# ─── Env state ────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    observation:       Observation
    cumulative_reward: float = 0.0
    done:              bool  = False
    info:              Dict[str, Any] = {}


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    decision:      DecisionType
    counter_price: Optional[float] = None   # custom price override for COUNTER_* actions
    reasoning:     Optional[str]   = None   # CoT field — logged and visible to judges


# ─── API wrappers ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    # Adding None as default makes the entire body optional
    task: str = "niche_alignment"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    metadata:    Dict[str, Any] = {}


# ─── Benchmark response ───────────────────────────────────────────────────────

class BenchmarkResponse(BaseModel):
    n_episodes:                 int
    mean_score:                 float
    std_dev:                    float
    best_episode:               float
    worst_episode:              float
    sellout_rate:               float
    burnout_rate:               float
    shady_brand_detected_rate:  float
    ambassador_unlocked:        bool
    integrity_test_passed_rate: float
    per_episode_scores:         List[float]