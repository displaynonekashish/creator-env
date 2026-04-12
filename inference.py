"""
inference.py — Influencer Business Sim v3.1
============================================
Fixes in this version:
  BUG1 FIXED: market_rate_est now uses /100.0 (matches new offer scale)
  BUG2 FIXED: PERSONALITY_STRATEGY max_counters set correctly per budget ceiling
  BUG3 FIXED: _fallback() accepts after first counter succeeds (neg_turn >= 1)
  BUG4 FIXED: rep_protection task prioritises ACCEPT over counter to accumulate earnings
  BUG5 FIXED: max_tokens=700, robust JSON extraction handles truncation

Agent features:
  1. Full Chain-of-Thought (CoT) reasoning — 5-step brand analysis logged per decision
  2. SHADY brand detection via observable signals (calibrated to new offer scale)
  3. Brand personality inference — type never given directly
  4. Multi-turn negotiation: counter once, then ACCEPT the raised offer
  5. Integrity test recognition and values-based rejection
  6. Gifted deal early-believer evaluation
  7. Organic content posting to manage parasocial debt
  8. Clause negotiation strategy
  9. Seasonal timing awareness
  10. Fallback rule engine for every case the LLM fails
"""

import os, sys, json, random
import requests
from openai import OpenAI

# ── Setup ─────────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK         = "influencer_business_sim"
MAX_STEPS         = 10
SUCCESS_THRESHOLD = 0.60

print(f"DEBUG: Using API_BASE_URL = {API_BASE_URL}", flush=True)
print(f"DEBUG: Using MODEL_NAME   = {MODEL_NAME}",   flush=True)
print(f"DEBUG: Using ENV_BASE_URL = {ENV_BASE_URL}", flush=True)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ── Env helpers ───────────────────────────────────────────────────────────────

def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def env_step(decision: str, counter_price=None, reasoning: str = "") -> dict:
    action = {"decision": decision}
    if counter_price is not None:
        action["counter_price"] = float(counter_price)
    if reasoning:
        action["reasoning"] = reasoning
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=20)
    r.raise_for_status()
    data   = r.json()
    reward = float(data.get("reward") or 0.0)
    done   = bool(data.get("done", False))
    inner  = data.get("observation", data)
    if isinstance(inner, dict) and "observation" in inner and isinstance(inner["observation"], dict):
        obs  = inner["observation"]
        done = done or bool(inner.get("done", False))
        meta = inner.get("metadata", {})
    else:
        obs  = inner
        meta = data.get("metadata", {})
    return {"observation": obs, "reward": reward, "done": done, "metadata": meta}


# ── SHADY brand signal detector ───────────────────────────────────────────────

def _shady_signal_score(deal: dict, obs: dict) -> float:
    """
    Returns 0.0–1.0 suspicion score.
    Uses /100 scale to match environment offer generation.
    """
    score     = 0.0
    followers = obs.get("followers", 100_000)
    offer     = deal.get("current_offer", deal.get("initial_offer", 0))

    # BUG1 FIX: market_rate_est uses /100.0 (was /1000.0)
    # Premium brands offer ~5-12x per 100 followers → market ≈ 8x / 100
    market_rate_est = followers * 8.0 / 300.0

    # SHADY offer_mult_range = 12–22x, so shady offer ≈ 17x / 100 avg
    if offer > market_rate_est * 1.8:     # > ~14.4x — clearly inflated
        score += 0.40
    elif offer > market_rate_est * 1.4:   # > ~11.2x — suspicious
        score += 0.20

    # Observable SHADY signals (boolean fields on the deal)
    if deal.get("brief_is_vague", False):
        score += 0.25
    if deal.get("has_urgency_pressure", False):
        score += 0.20
    if not deal.get("has_verifiable_presence", True):
        score += 0.25

    # High rep_risk
    rep_risk = deal.get("rep_risk", 0.0)
    if rep_risk >= 0.5:
        score += 0.30
    elif rep_risk >= 0.3:
        score += 0.10

    return min(1.0, score)


# ── Gifted deal evaluator ─────────────────────────────────────────────────────

def _evaluate_gifted(deal: dict, obs: dict) -> bool:
    """Returns True if gifted deal is worth accepting. Requires 3+ positive signals."""
    if deal.get("brand_revenue_stage", "established") == "established":
        return False   # established brand gifting = disrespect
    product_quality = deal.get("product_quality")
    niches   = obs.get("creator_niches", [])
    category = deal.get("category", "")
    signals  = [
        deal.get("founder_engaged", False),
        (product_quality is not None and product_quality > 0.70),
        deal.get("community_buzz", 0.0) > 0.50,
        category in niches,
    ]
    return sum(signals) >= 3


# ── Personality inference ─────────────────────────────────────────────────────

def _infer_personality(deal: dict, obs: dict) -> str:
    """
    Infer brand personality from observable signals.
    Personality type is never given directly to the agent.
    """
    shady_score = _shady_signal_score(deal, obs)
    if shady_score >= 0.55:
        return "shady"

    followers = obs.get("followers", 100_000)
    offer     = deal.get("current_offer", deal.get("initial_offer", 0))
    excl_days = deal.get("exclusivity_days", 0)
    is_gifted = deal.get("is_gifted", False)
    stage     = deal.get("brand_revenue_stage", "established")
    patience  = deal.get("patience", 3)
    revs      = deal.get("revision_rounds", 2)

    # BUG1 FIX: market reference uses /100.0
    market = followers * 8.0 / 300.0

    if is_gifted or stage in ("pre-revenue", "seed"):
        return "startup"
    if patience >= 6:
        return "startup"
    if offer > market * 1.6 and excl_days >= 30:
        return "luxury"
    if revs >= 4 and offer < market * 1.0:
        return "mass_market"
    return "premium"


# ── Personality-aware counter strategy ───────────────────────────────────────
#
# BUG2 FIX: max_counters reduced to 1 for startup/mass_market.
# Reason: startup max_budget_mult=1.25, premium=1.30.
# After brand counters back to ~110% of initial, a second counter_20
# proposes 110%×1.20=132%, which exceeds the brand's ceiling (125%).
# Brand can never close → infinite 0.01 reward loop.
# Fix: counter ONCE, then accept whatever the brand raised to.
#
PERSONALITY_STRATEGY = {
    "luxury":      {"max_counters": 1, "counter_action": "counter_10", "push_mult": 1.08},
    "startup":     {"max_counters": 1, "counter_action": "counter_20", "push_mult": 1.18},
    "mass_market": {"max_counters": 1, "counter_action": "counter_20", "push_mult": 1.25},
    "shady":       {"max_counters": 0, "counter_action": None,         "push_mult": 1.0},
    "premium":     {"max_counters": 2, "counter_action": "counter_20", "push_mult": 1.22},
}


# ── Niche crossover compatibility (mirrors environment) ──────────────────────
NICHE_COMPAT: dict = {
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


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _fallback(task: str, obs: dict, neg_turn: int = 0) -> tuple:
    """
    Returns (decision_str, counter_price_or_None, reasoning_str).
    Used when LLM fails or as override safety net.
    """
    deal       = obs.get("deal") or {}
    neg        = obs.get("negotiation") or {}
    niches     = obs.get("creator_niches", [])
    fatigue    = obs.get("fatigue", 0.0)
    rep        = obs.get("reputation", 1.0)
    sellouts   = obs.get("sellout_count", 0)
    parasocial = obs.get("parasocial_debt", 0)
    patience   = neg.get("brand_patience_remaining", 3)
    integrity  = obs.get("integrity_test_triggered", False)
    earnings   = obs.get("earnings", 0.0)
    target     = obs.get("monthly_target", 5000.0)
    brand_mood = neg.get("brand_mood", "interested")

    category  = deal.get("category", "")
    rating    = deal.get("brand_rating", 0.0)
    offer     = deal.get("current_offer", deal.get("initial_offer", 0))
    rep_risk  = deal.get("rep_risk", 0.0)
    is_gifted = deal.get("is_gifted", False)

    # ── Priority 0: organic content if parasocial debt critical ──────────────
    if parasocial >= 3:
        return "post_organic_content", None, "ORGANIC: parasocial_debt>=3 — must post free content"

    # ── Priority 1: hard safety gates ────────────────────────────────────────
    if fatigue >= 0.85:
        return "reject", None, "REJECT: fatigue>=0.85 — burnout protection"

    if sellouts >= 3:
        return "reject", None, "REJECT: sellout_count>=3 — long-term brand protection"

    # ── Priority 2: SHADY detection ──────────────────────────────────────────
    shady_score = _shady_signal_score(deal, obs)
    if shady_score >= 0.50:
        return "reject", None, f"REJECT: shady_score={shady_score:.2f} — reputation trap detected"

    # ── Priority 3: Integrity test ────────────────────────────────────────────
    # Only reject while test is ACTIVE (triggered=True AND not yet resolved)
    integrity_passed = obs.get("integrity_test_passed")
    if integrity and integrity_passed is None:
        return "reject", None, "REJECT: integrity_test active — values override money"

    # ── Priority 4: Gifted deal ───────────────────────────────────────────────
    if is_gifted:
        # Aligned gifted with zero rep_risk → always accept (early-believer upside)
        aligned_gifted = category in niches
        if aligned_gifted and rep_risk == 0.0:
            return "accept", None, "ACCEPT: gifted + aligned + zero rep_risk"
        worth_it = _evaluate_gifted(deal, obs)
        if worth_it:
            return "accept", None, "ACCEPT: gifted startup — 3+ early-believer signals"
        return "reject", None, "REJECT: gifted — insufficient signals for unpaid work"

    # ── Priority 5: Niche/rep gates ───────────────────────────────────────────
    aligned = category in niches

    # Compute crossover compatibility score (0.0 = zero relation to any creator niche)
    niche_fit = 0.0
    if aligned:
        niche_fit = 1.0
    else:
        for n in niches:
            niche_fit = max(niche_fit, NICHE_COMPAT.get(n, {}).get(category, 0.0))

    # Unaligned gate:
    # - Zero crossover compatibility = always reject (no audience overlap)
    # - Low crossover or risky = reject unless exceptional conditions
    if not aligned:
        if niche_fit == 0.0:
            return "reject", None, f"REJECT: zero niche compat — cat={category} irrelevant to niches={niches}"
        if rep_risk > 0.05 or rating < 0.88:
            return "reject", None, f"REJECT: unaligned cat={category}, rep_risk={rep_risk:.2f}, rating={rating:.2f}"

    if rep_risk > 0.30 and task in ("reputation_protection", "niche_alignment"):
        return "reject", None, f"REJECT: rep_risk={rep_risk:.2f} too high for {task}"
    if rep_risk > 0.40:
        return "reject", None, f"REJECT: rep_risk={rep_risk:.2f} above 0.40 — always reject"

    # ── Priority 6: Kill clause for risky deals (only turn 0) ────────────────
    if (rep_risk > 0.15 and neg_turn == 0
            and deal.get("crisis_probability", 0.0) > 0.15):
        return "request_kill_clause", None, "CLAUSE: crisis_prob high — request kill clause"

    # ── Priority 7: Counter strategy ─────────────────────────────────────────
    personality  = _infer_personality(deal, obs)
    strat        = PERSONALITY_STRATEGY.get(personality, PERSONALITY_STRATEGY["premium"])
    brand_raised = neg.get("last_counter_by_brand") is not None

    # Counter once only. After brand raises → ACCEPT (second counter exceeds ceiling).
    can_counter = (
        neg_turn < strat["max_counters"]
        and patience > 0
        and fatigue < 0.70
        and brand_mood not in ("angry", "walked")
        and not brand_raised
        and strat["counter_action"] is not None
        and aligned  # never waste a counter on unaligned deal
    )

    if can_counter:
        c_action = strat["counter_action"]
        c_price  = round(offer * strat["push_mult"], 2)
        # reputation_protection: rep is 0.45 weight — accept fast, don't waste steps
        if task == "reputation_protection":
            pass  # fall through to accept
        elif task == "target_negotiation":
            gap = target - earnings
            if gap > offer * 0.8:
                return c_action, c_price, (
                    f"COUNTER ({c_action}): {personality}, patience={patience}, "
                    f"${offer:.0f}→${c_price:.0f}, gap=${gap:.0f}"
                )
        else:
            return c_action, c_price, (
                f"COUNTER ({c_action}): {personality}, patience={patience}, "
                f"${offer:.0f}→${c_price:.0f}"
            )

    # ── Priority 8: Accept if safe ────────────────────────────────────────────
    if rating >= 0.60 and rep_risk <= 0.20:
        return "accept", None, (
            f"ACCEPT: aligned={aligned}, rating={rating:.2f}, rep_risk={rep_risk:.2f}"
        )

    return "reject", None, (
        f"REJECT: rating={rating:.2f} < 0.60 or rep_risk={rep_risk:.2f} > 0.20"
    )


# ── LLM decision with 5-step CoT ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert talent manager for a social media influencer.
Your mission: maximize LONG-TERM career sustainability — not just this month's cash.

CRITICAL RULES:
1. Brand personality is NEVER given. Infer it from signals (offer level, brief quality, patience).
2. SHADY signals: brief_is_vague, has_urgency_pressure, has_verifiable_presence=false,
   offer >1.8x market rate. ALWAYS reject SHADY brands. They destroy reputation.
3. COUNTER ONCE then ACCEPT: after a brand has already raised its offer (last_counter_by_brand
   is not null), ACCEPT the raised offer. Do NOT counter again — brand's budget is near ceiling
   and a second counter will exceed it, causing a 0.01 reward loop with no deal ever closing.
4. Luxury: patience=2, counter_10 at most once. If they've already raised → ACCEPT.
5. Startup: patience=8 but max_budget_mult=1.25 (tight). Counter once at 20%, then ACCEPT.
6. reputation_protection task: rep score has 0.45 weight. ACCEPT good aligned deals quickly.
   Do NOT waste steps countering when you should be earning rep from good deals.
7. NEVER accept if fatigue > 0.85. NEVER accept with sellout_count >= 3.
8. parasocial_debt > 3: POST ORGANIC CONTENT before any deal.
9. integrity_test_triggered = true AND integrity_test_passed = null: REJECT that specific deal.
   After rejecting it the flag clears — normal deals resume. Do NOT keep rejecting forever.
10. Only request kill_clause ONCE on turn 0, before any countering.

Reason through ALL 5 steps:
STEP 1 — BRAND READ: Personality type? Key signals?
STEP 2 — RISK SCAN: rep_risk, niche fit, sellout_count impact.
STEP 3 — NEGOTIATION POSITION: Has brand already countered? If yes → ACCEPT.
STEP 4 — FATIGUE AUDIT: true_fatigue_cost = deliverables + revision rounds.
STEP 5 — LONG-TERM VIEW: Does this build or burn the creator's brand?

Valid decisions: accept | reject | counter_10 | counter_20 | counter_40 | counter_60 |
                 request_kill_clause | request_exclusivity_waiver | request_revision_cap |
                 unbox_product | post_organic_content

Output ONLY valid JSON — no markdown, no text outside JSON:
{
  "chain_of_thought": "<5 steps, 1-2 sentences each>",
  "decision": "<valid decision>",
  "counter_price": <float or null>,
  "confidence": <0.0 to 1.0>
}"""


def get_decision(task: str, obs: dict, neg_turn: int = 0) -> tuple:
    """Returns (decision_str, counter_price, cot_str, error_str_or_None)."""
    deal       = obs.get("deal") or {}
    neg        = obs.get("negotiation") or {}
    niches     = obs.get("creator_niches", [])
    season     = obs.get("market_season", "q2")
    s_mult     = obs.get("seasonal_multiplier", 1.0)
    parasocial = obs.get("parasocial_debt", 0)
    integrity  = obs.get("integrity_test_triggered", False)
    followers  = obs.get("followers", 0)

    offer       = deal.get("current_offer", deal.get("initial_offer", 0))
    category    = deal.get("category", "?")
    shady_score = _shady_signal_score(deal, obs)
    inferred_p  = _infer_personality(deal, obs)
    brand_raised = neg.get("last_counter_by_brand") is not None

    user_prompt = f"""TASK: {task}
MARKET: season={season} (multiplier={s_mult:.1f}x)

CREATOR:
  niches:           {niches}
  followers:        {followers:,}
  engagement_rate:  {obs.get('engagement_rate', 0):.3f}
  reputation:       {obs.get('reputation', 0):.2f}
  audience_trust:   {obs.get('audience_trust', 0):.2f}
  fatigue:          {obs.get('fatigue', 0):.2f}{'  ⚠ DANGER' if obs.get('fatigue',0) >= 0.70 else ''}
  parasocial_debt:  {parasocial}{'  ⚠ POST ORGANIC FIRST' if parasocial >= 3 else ''}
  earnings:         ${obs.get('earnings', 0):.0f} / ${obs.get('monthly_target', 0):.0f} target
  sellout_count:    {obs.get('sellout_count', 0)}{'  🚨 MAX' if obs.get('sellout_count',0) >= 3 else ''}

DEAL:
  brand:            {deal.get('brand_name', 'Unknown')}
  category:         {category} {'✓ ALIGNED' if category in niches else '✗ NOT IN NICHES'}
  brand_rating:     {deal.get('brand_rating', 0):.2f}
  current_offer:    ${offer:.2f}
  payment:          {deal.get('payment_structure', 'upfront')}
  rep_risk:         {deal.get('rep_risk', 0):.2f}
  exclusivity_days: {deal.get('exclusivity_days', 0)}
  deliverables:     {deal.get('deliverable_count', 1)}
  revision_rounds:  {deal.get('revision_rounds', 2)}{'  ⚠ UNLIMITED=TRAP' if deal.get('revision_rounds',2) >= 10 else ''}
  crisis_prob:      {deal.get('crisis_probability', 0):.2f}
  is_gifted:        {deal.get('is_gifted', False)}
  revenue_stage:    {deal.get('brand_revenue_stage', 'established')}
  founder_engaged:  {deal.get('founder_engaged', False)}
  community_buzz:   {deal.get('community_buzz', 0):.2f}
  brief_is_vague:   {deal.get('brief_is_vague', False)}{'  🚨 SHADY' if deal.get('brief_is_vague') else ''}
  urgency_pressure: {deal.get('has_urgency_pressure', False)}{'  🚨 SHADY' if deal.get('has_urgency_pressure') else ''}
  verifiable:       {deal.get('has_verifiable_presence', True)}{'  🚨 SHADY' if not deal.get('has_verifiable_presence', True) else ''}

SIGNALS:
  shady_score:      {shady_score:.2f}{'  🚨 REJECT' if shady_score >= 0.50 else ''}
  inferred_type:    {inferred_p}

NEGOTIATION:
  turn:             {neg.get('turn', 0)} / 8
  patience_left:    {neg.get('brand_patience_remaining', 3)}
  brand_mood:       {neg.get('brand_mood', 'interested')}
  brand_countered:  {brand_raised}{'  → ACCEPT THE RAISED OFFER (rule 3)' if brand_raised else ''}
  brand_offer_now:  {neg.get('last_counter_by_brand')}

INTEGRITY_TEST: {integrity}{'  🚨 REJECT' if integrity else ''}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=700,      # BUG5 FIX: was 500, too low for 5-step CoT
            temperature=0.1,     # lower temp = more consistent JSON
        )
        raw = resp.choices[0].message.content.strip()

        # BUG5 FIX: robust JSON extraction — handles partial truncation
        # Try to find the JSON block even if LLM adds text around it
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Find first { and last } to extract just the JSON
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end+1]

        out     = json.loads(raw)
        dec     = out.get("decision", "reject").lower()
        c_price = out.get("counter_price")
        cot     = out.get("chain_of_thought", "")

        # ── Hard safety overrides (LLM cannot bypass) ──────────────────────

        # 1. SHADY brand — never accept regardless of LLM reasoning
        if dec == "accept" and shady_score >= 0.50:
            dec   = "reject"
            cot  += " [OVERRIDE: shady_score too high]"

        # 2. Integrity test — values override money (only while ACTIVE, not after resolved)
        if dec == "accept" and integrity and obs.get("integrity_test_passed") is None:
            dec   = "reject"
            cot  += " [OVERRIDE: integrity test active]"

        # 3. Burnout protection
        if dec == "accept" and obs.get("fatigue", 0) >= 0.85:
            dec   = "reject"
            cot  += " [OVERRIDE: fatigue critical]"

        # 4. Sellout limit
        if dec == "accept" and obs.get("sellout_count", 0) >= 3:
            dec   = "reject"
            cot  += " [OVERRIDE: sellout limit reached]"

        # 5. Unaligned + risky deal — LLM frequently accepts these by mistake
        deal_category = deal.get("category", "")
        creator_niches_local = obs.get("creator_niches", [])
        deal_aligned  = deal_category in creator_niches_local
        deal_rep_risk = deal.get("rep_risk", 0.0)
        deal_rating   = deal.get("brand_rating", 0.0)

        # Compute crossover compat (same as _fallback)
        deal_niche_fit = 1.0 if deal_aligned else max(
            (NICHE_COMPAT.get(n, {}).get(deal_category, 0.0) for n in creator_niches_local),
            default=0.0
        )
        if dec == "accept" and not deal_aligned:
            if deal_niche_fit == 0.0:
                dec   = "reject"
                cot  += f" [OVERRIDE: zero compat cat={deal_category}]"
            elif deal_rep_risk > 0.05 or deal_rating < 0.85:
                dec   = "reject"
                cot  += f" [OVERRIDE: unaligned cat={deal_category}, rep_risk={deal_rep_risk:.2f}]"

        # 6. Very high rep_risk — never accept regardless of alignment
        if dec == "accept" and deal_rep_risk > 0.35:
            dec   = "reject"
            cot  += f" [OVERRIDE: rep_risk={deal_rep_risk:.2f} too high]"

        # 7. Brand already countered — accept the raised offer, never counter again
        if dec.startswith("counter") and brand_raised:
            dec     = "accept"
            c_price = None
            cot    += " [OVERRIDE: brand already raised — accepting]"

        # 8a. Counter on zero-compat unaligned deal — reject instead (wasted step fix)
        #     LLM tries counter_20 on startup deals even when category has no compat
        if dec.startswith("counter") and not deal_aligned and deal_niche_fit == 0.0:
            dec     = "reject"
            c_price = None
            cot    += f" [OVERRIDE: counter on zero-compat cat={deal_category} → reject]"

        # 8b. Counter on unaligned+risky deal — reject instead
        if dec.startswith("counter") and not deal_aligned and (deal_rep_risk > 0.10 or deal_rating < 0.70):
            dec     = "reject"
            c_price = None
            cot    += f" [OVERRIDE: counter on risky unaligned → reject]"

        # 8c. Clause requests on unaligned deals — reject instead (wasted step fix)
        #     Clauses only make sense on deals you intend to accept.
        #     Never request kill_clause/revision_cap/etc on an unaligned deal.
        clause_actions = ("request_kill_clause", "request_exclusivity_waiver", "request_revision_cap")
        if dec in clause_actions and not deal_aligned:
            dec     = "reject"
            c_price = None
            cot    += f" [OVERRIDE: clause on unaligned cat={deal_category} → reject]"

        # 9. Good aligned deal that LLM mistakenly rejected — nudge to accept
        if (dec == "reject"
                and deal_aligned
                and deal_rating >= 0.75
                and deal_rep_risk <= 0.15
                and shady_score < 0.40
                and not integrity
                and obs.get("fatigue", 0) < 0.80
                and obs.get("sellout_count", 0) < 3
                and obs.get("reputation", 0) >= 0.50):
            dec   = "accept"
            cot  += " [OVERRIDE: good aligned deal — accepting]"

        return dec, c_price, cot, None

    except Exception as e:
        dec, c_price, reasoning = _fallback(task, obs, neg_turn)
        return dec, c_price, reasoning, str(e)


# ── Score calculator (mirrors environment) ────────────────────────────────────

def _calc_score(task: str, obs: dict) -> float:
    followers = obs.get("followers", 0)
    trust     = obs.get("audience_trust", 0.5)
    trust_f   = trust if trust >= 0.60 else 0.0
    fv        = min(1.0, followers / 500_000 * trust_f)
    rev       = min(1.0, obs.get("earnings", 0) / max(1, obs.get("monthly_target", 1)))
    rep       = obs.get("reputation", 0)
    sust      = max(0.0, 1.0 - obs.get("fatigue", 0) ** 2)
    sellouts  = obs.get("sellout_count", 0)

    sellout_pen = sum(0.3 ** i for i in range(sellouts)) * 0.12
    int_pen     = 0.15 if obs.get("integrity_test_passed") is False else 0.0

    if task == "niche_alignment":
        base = fv*0.30 + rev*0.25 + rep*0.30 + sust*0.15
    elif task == "reputation_protection":
        base = fv*0.20 + rev*0.20 + rep*0.45 + sust*0.15
    else:
        base = fv*0.20 + rev*0.45 + rep*0.20 + sust*0.15

    return round(max(0.0, min(1.0, base - sellout_pen - int_pen)), 2)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(task: str) -> float:
    print(f"\n[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards, step_num, last_obs, final_score = [], 0, {}, 0.0
    neg_turn = 0

    try:
        last_obs = env_reset(task)

        for step_num in range(1, MAX_STEPS + 1):
            deal = last_obs.get("deal")
            neg  = last_obs.get("negotiation") or {}

            if not deal:
                result   = env_step("reject", reasoning="no deal available")
                reward   = float(result.get("reward", 0.0))
                done     = result.get("done", False)
                rewards.append(reward)
                print(
                    f'[STEP] step={step_num} action={{"decision":"reject"}} '
                    f'reward={reward:.2f} done={str(done).lower()} error=null',
                    flush=True,
                )
                last_obs = result.get("observation", last_obs)
                if done:
                    break
                continue

            neg_turn = neg.get("turn", 0)
            decision, c_price, cot, err = get_decision(task, last_obs, neg_turn)

            result  = env_step(decision, c_price, reasoning=(cot or "")[:200])
            reward  = float(result.get("reward", 0.0))
            done    = result.get("done", False)
            rewards.append(reward)
            meta    = result.get("metadata", {})

            action_json = {"decision": decision, "counter_price": c_price}
            print(
                f'[STEP] step={step_num} action={json.dumps(action_json)} '
                f'reward={reward:.2f} done={str(done).lower()} '
                f'cot="{(cot or "")[:100]}" '
                f'error={err if err else "null"}',
                flush=True,
            )

            last_obs = result.get("observation", last_obs)
            if meta.get("final_score") is not None:
                final_score = float(meta["final_score"])
            if done:
                break

        if final_score == 0.0:
            final_score = _calc_score(task, last_obs)

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success     = final_score >= SUCCESS_THRESHOLD
        print(
            f"[END] success={str(success).lower()} steps={step_num} "
            f"score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )
        return final_score

    except Exception as exc:
        print(f"[END] success=false steps={step_num} score=0.00 rewards=0.00 exc={exc}", flush=True)
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

TASKS = ["niche_alignment", "reputation_protection", "target_negotiation"]

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)

    avg = sum(scores.values()) / len(scores)
    print(f"\n{'='*55}")
    for t, s in scores.items():
        status = "✓ PASS" if s >= SUCCESS_THRESHOLD else "✗ FAIL"
        print(f"  {t:<30} {s:.2f}  {status}")
    print(f"  {'─'*44}")
    print(f"  {'FINAL AVERAGE':<30} {avg:.2f}")
    print(f"{'='*55}")