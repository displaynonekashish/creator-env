---
title: Influencer Sim
emoji: 🔹
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# Influencer Business Sim
### *The creator economy deal room, simulated with real-world precision*

> **Meta PyTorch OpenEnv Hackathon Submission** | v3.0.0 | Phase 2 Passed 

---

##  The Core Thesis

The creator economy generates **$250B+ annually**. Yet the decision-making at its core — which brand deals to accept, how to negotiate terms, when to walk away — is still done by gut feel, spreadsheets, or expensive talent agencies.

This environment simulates that decision space with clinical precision. An agent that masters this environment could, realistically, **outperform a junior talent manager on deal evaluation**.

### Why it's hard

> Most negotiation environments reward greedy agents. **This one punishes them.**

The highest-value offer in the observation is statistically the most likely reputation trap (the **SHADY** brand archetype). An agent that sorts by offer size will reliably destroy its creator's reputation within 3 episodes. Solving this environment requires social intelligence, not math.

---

## 🏗️ Architecture

```
creator-env/
├── inference.py                      # Baseline agent (hackathon entry point)
├── openenv.yaml                      # Registry Config & Metadata
├── Dockerfile                        # HF Spaces deployment (port 7860)
├── requirements.txt
└── server/
    ├── app.py                        # FastAPI server (OpenEnv spec)
    ├── models.py                     # Pydantic data structures (all types)
    └── creator_env_environment.py    # Environment brain (all mechanics)
The environment operates on a standard RL loop, optimized for high-fidelity social reasoning:
Observation (Market/Deal) → LLM Reasoning (5-Step CoT) → Action (Negotiate/Post) → Reward (Long-term Equity).
```

**Stack:** FastAPI · Pydantic v2 · OpenAI SDK · Docker · Hugging Face Spaces  
**Spec:** OpenEnv (`/reset`, `/step`, `/state`)  
**LLM:** Routes through LiteLLM proxy via `API_BASE_URL` + `API_KEY` env vars

---

##  Real-World Grounding

Every mechanic maps to how the creator economy actually works:

| Environment Mechanic | Real-World Counterpart |
|---|---|
| Brand patience meter | Brands pull offers if you ghost them |
| SHADY brand signals | Inflated offers + vague briefs are real scams |
| Exclusivity lockout | Standard clause in all CPG/fashion contracts |
| Revision cap negotiation | Agencies cap at 2 rounds as standard |
| Payment structure choice | Upfront vs rev-share is creator's biggest risk call |
| Kill clause mechanic | Standard protection in all agency contracts |
| Q4 budget multiplier | 60% of influencer budgets deploy Oct–Dec |
| Gifted early-believer | Stanley Cup (2022), Glossier — being early = 3–5x return |
| Parasocial debt | Creators must post free content to maintain audience loyalty |
| Integrity test | Every creator eventually gets offered money to betray their niche |
| Media Kit system | Brands check follower/engagement data before making offers |
| Manager NPC at 100K | Most mid-tier creators sign management at this threshold |
| Platform format shift | Reels launch (2021) changed creator economics overnight |

---

##  Environment Mechanics

### Action Space (11 actions)

| Action | Description |
|---|---|
| `accept` | Sign the deal |
| `reject` | Pass on the deal |
| `counter_10` | Counter at +10% — safe for luxury brands |
| `counter_20` | Counter at +20% — standard push |
| `counter_40` | Counter at +40% — aggressive |
| `counter_60` | Counter at +60% — makes luxury/shady angry |
| `request_kill_clause` | Protect against brand PR crises |
| `request_exclusivity_waiver` | Negotiate out the lockout clause |
| `request_revision_cap` | Cap revision rounds at 2 |
| `unbox_product` | Reveal hidden product quality (needed for gifted deal eval) |
| `post_organic_content` | Reduce parasocial debt, restore audience engagement |

### Observation Space (14 dimensions)

```python
creator_niches           # List[str] — up to 3, with crossover compatibility matrix
niche_reputation         # Dict[str, float] — per-niche score 0–1
followers                # int — grows/shrinks based on decisions
engagement_rate          # float — degrades with fatigue + parasocial debt
media_kit                # MediaKit — outdated = lowball offers
reputation               # float [0,1]
audience_trust           # float [0,1] — <0.6 makes followers worthless in score
fatigue                  # float [0,1] — non-linear cliff at 0.7 and 0.9
parasocial_debt          # int — >3 triggers engagement drop
consecutive_bad_deals    # int — ≥3 triggers trust crisis (-20% follower unfollow)
market_season            # MarketSeason — 0.60x (Jan) to 1.50x (Nov Black Friday)
lockout_calendar         # Dict[str, int] — category exclusivity days remaining
deal                     # BrandDeal | null — full contract terms
negotiation              # NegotiationState — turn, patience, mood, clauses granted
```

---

## Brand Personality System

**The agent is NEVER told the personality type. It must infer from observable signals.**

| Personality | Patience | Budget Range | Key Signals | Trap? |
|---|---|---|---|---|
| **LUXURY** | 2 turns | High (1.8–3.5x) | Precise brief, exclusivity required | No |
| **STARTUP** | 8 turns | Low (0.3–0.9x) | Founder engaged, early revenue stage | No |
| **MASS MARKET** | 4 turns | Medium (0.8–1.6x) | Many deliverables, wants virality | No |
| **PREMIUM** | 3 turns | Medium (1.2–2.2x) | Balanced terms, good rating | No |
| **SHADY**  | 3 turns | Very High (2.5–5.0x) | Vague brief, urgency pressure, no website | **YES** |

### SHADY Brand — The Core Trap

```
Observable signals the agent must learn to read:
  brief_is_vague          = True   → "just be authentic!"
  has_urgency_pressure    = True   → "offer expires in 1 hour"
  has_verifiable_presence = False  → no company website found
  offer                   > 2.5x market rate  → suspiciously inflated

If accepted:  reputation -= 0.55, audience_trust -= 0.35, crisis_probability = 0.40
If rejected:  reputation += 0.03, audience_trust += 0.04, shady_detected += 1
```

---

## Multi-Turn Negotiation (up to 8 turns)

```
Brand Mood State Machine:
  INTERESTED → COOLING → DESPERATE (patience=1, push harder!)
                       → ANGRY     (counter_60 used, brand may walk)
                       → WALKED    (patience exhausted)

Anchoring bonus: agent who counters first gets +edge on final accepted price
Brand counter-offer decay: each round, brand raises less (counter_decay × turn)
Clause negotiation: kill_clause, exclusivity_waiver, revision_cap mid-negotiation
```

---

## Gifted Deal / Early-Believer System

Not all gifted deals are bad. Context determines everything.

```python
# Established brand gifting = disrespect (they have budget) → REJECT

# Pre-revenue / seed startup — evaluate 4 signals:
signals = [
    founder_engaged,           # Did the founder personally reach out?
    product_quality > 0.70,    # Revealed only via UNBOX_PRODUCT action
    community_buzz > 0.50,     # Organic mentions trending?
    category in creator_niches # Genuine niche fit?
]
# Accept if 3+ signals positive → early-believer relationship

# If brand later goes viral:
#   followers *= 1.15
#   audience_trust += 0.12
#   reputation += 0.08
#   brand_relationship = "early_believer"  → loyalty premium on future deals
```

**Real-world basis:** Stanley Cup gifted to creators in 2022. Glossier gave early creators equity. Being early has compounding value.

---

##  The Integrity Test

**Once per episode. The moment where math cannot tell you what to do.**

```
A FastFeast (mass-market fast food) deal appears with offer = 3x market rate.
Creator has stated values: ["body_positivity", "no_alcohol", "sustainability"]

If ACCEPTED:
  audience_trust     -= 0.35   (permanent, not recoverable)
  niche_reputation["wellness"] -= 0.40
  integrity_test_passed = False → penalty of -0.15 in final score

If REJECTED:
  reputation         += 0.15
  audience_trust     += 0.08
  Unlocks "values-aligned" brand tier — ethical brands now seek creator out
```

The agent that can reason "my creator built her audience on wellness — accepting this tells them she was lying" is doing genuine **moral reasoning in an RL environment**.

---

##  Fatigue Non-Linear Cliff

```python
fatigue < 0.50  →  full performance
fatigue < 0.70  →  engagement_rate *= 0.95 (slight decline)
fatigue < 0.90  →  engagement_rate *= 0.88 (significant decline)
fatigue >= 1.0  →  BURNOUT — episode ends immediately, score = 0.0
```

Fatigue costs per action:
- `accept` → `deal.true_fatigue_cost` (deliverables × revision_rounds)
- `counter_*` → +0.04 per turn
- `post_organic_content` → +0.03 (costs time, not money)
- `unbox_product` → +0.01

**Unlimited revision rounds (revision_rounds=99) = hidden trap** — true_fatigue_cost calculation exposes it.

---

##  Parasocial Debt

Creators who only post ads lose audience loyalty. The agent must post organic content.

```python
parasocial_debt += 1  # on every sponsored deal accepted
# > 3: engagement_rate *= 0.88  # audience feels used

post_organic_content:
    parasocial_debt -= 2          # reset debt
    audience_trust  += 0.03       # "she still shows up for us"
    engagement_rate *= 1.05       # audience re-engages
```

No other submission will have this constraint. It forces the agent to **budget non-revenue actions** — a real and important creator decision.

---

##  Seasonal Deal Calendar

```
Season,Month,Multiplier,Strategy
Q1,January,0.60x,Post-holiday budget freeze; Post Organic Content
Q1,February,0.80x,Valentine's Day push; Selective acceptance
Q2,Mar - Jun,0.90x,Steady state; Build brand relationships
Q3,July,0.75x,Summer slump; Best time for unboxing/rest
Q4,October,1.30x,Holiday ramp-up; High-value deal influx
Q4,November,1.50x,Black Friday Peak; Save fatigue for this!
Q4,December,1.40x,Christmas push; Final year-end budget burn

**Strategy:** A smart agent conserves fatigue bandwidth in Q3 to be available for Q4 peak. 60% of real influencer budgets deploy October–December.

---

##  Payment Structures

| Structure | Certainty | Multiplier | Expected Value |
|---|---|---|---|
| `upfront` | 1.00 | 1.00 | offer × 1.00 |
| `milestone` | 0.85 | 1.15 | offer × 0.98 |
| `revenue_share` | 0.30 | 2.50 | offer × 0.75 (high variance) |
| `gifted_only` | 1.00 | 0.00 | $0 cash (relationship value only) |

---

##  Task Suite

| Task | Difficulty | Objective | Key Challenge |
|---|---|---|---|
| **niche_alignment** | Easy | Filter deals by niche relevance | Multi-niche crossover matrix |
| **reputation_protection** | Medium | Strict brand-safety gatekeeping | SHADY trap + integrity test |
| **target_negotiation** | Hard | Hit $6,250 revenue via negotiation | All personalities + seasonal timing |

---

##  Scoring Function

**No immediate cash rewards. Sparse reward at episode end.**

```python
# Followers only count if audience trusts the creator
follower_value = followers × (audience_trust if audience_trust >= 0.60 else 0.0)

# Sustainability is non-linear — burnout is catastrophic
sustainability = max(0, 1 - fatigue²)

# Base score (weights vary by task)
base = follower_value×W1 + revenue×W2 + reputation×W3 + sustainability×W4 + brand_diversity×W5

# Exponential sellout penalty — 3 sellouts is devastating
sellout_penalty = Σ(0.3^i for i in range(sellout_count)) × 0.12

# Values failure
integrity_penalty = 0.15 if integrity_test_failed else 0.0

final_score = base - sellout_penalty - integrity_penalty
```

**Success threshold: 0.60** across all tasks.

---

##  Agent Design (inference.py)

The baseline agent uses a **hybrid deterministic + LLM** architecture:

### 5-Step Chain-of-Thought Reasoning

```
STEP 1 — BRAND READ:       What personality type? What signals suggest it?
STEP 2 — RISK SCAN:        Rep risk, niche fit, audience trust impact.
STEP 3 — NEGOTIATION POS.: BATNA? Anchor with counter? Brand mood?
STEP 4 — FATIGUE AUDIT:    True cost = deliverables + revision rounds.
STEP 5 — LONG-TERM VIEW:   Builds relationships or burns them?
```

### SHADY Detection (rule-based, never trusts LLM alone)

```python
shady_score = 0.0
if offer > market_rate_estimate × 2.5:  score += 0.40
if brief_is_vague:                       score += 0.25
if has_urgency_pressure:                 score += 0.20
if not has_verifiable_presence:          score += 0.25
if rep_risk >= 0.5:                      score += 0.30

if shady_score >= 0.50: → REJECT (hard override, LLM cannot override this)
```

### Safety Overrides (LLM cannot bypass)
- SHADY score ≥ 0.50 → always reject
- Integrity test active → always reject
- Fatigue ≥ 0.85 → always reject
- Sellout count ≥ 3 → always reject

---

## Setup & Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start environment server
python server/app.py

# In another terminal, run the agent
python inference.py
```

### Environment Variables

```bash
API_KEY=your_llm_key
API_BASE_URL=https://your-litellm-proxy/v1
MODEL_NAME=meta-llama/Llama-3-8b-instruct
ENV_BASE_URL=http://localhost:7860   # or HF Space URL
```

### Docker (Hugging Face Spaces)

```bash
docker build -t creator-env .
docker run -p 7860:7860 \
  -e API_KEY=... \
  -e API_BASE_URL=... \
  -e MODEL_NAME=... \
  creator-env
```

---

##  API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task": "niche_alignment"}` |
| `/step` | POST | Submit action. Body: `{"action": {"decision": "counter_20", "reasoning": "..."}}` |
| `/state` | GET | Full internal environment state |
| `/benchmark` | GET | Run N episodes, return score distribution |
| `/health` | GET | Health check |
OpenEnv Compliance
This environment is fully headless and stateless via the FastAPI wrapper. This architecture ensures the simulation is:

Vectorization Ready: Supports parallel training across multiple instances.

Judge-Verified: Passes all openenv validate automated checks.

Cloud Native: Ready for one-click deployment to Hugging Face Spaces or Docker-based clusters.

### `/benchmark` Response

```json
{
  "n_episodes": 10,
  "mean_score": 0.72,
  "std_dev": 0.09,
  "best_episode": 0.88,
  "worst_episode": 0.51,
  "sellout_rate": 0.12,
  "burnout_rate": 0.20,
  "shady_brand_detected_rate": 0.65,
  "ambassador_unlocked": false,
  "integrity_test_passed_rate": 0.80,
  "per_episode_scores": [0.71, 0.88, 0.64, ...]
}
```

---

##  Baseline Scores

Task,Score, Status
niche_alignment, 0.75, PASS
reputation_protection, 0.92, PASS
target_negotiation, 0.86, PASS
FINAL AVERAGE, 0.84, PHASE 2 COMPLETE

---

##  Complexity Features

```yaml
- multi_turn_negotiation_8_turns
- shady_brand_trap_detection
- 5_brand_personalities_hidden_from_agent
- brand_mood_state_machine
- gifted_early_believer_system
- parasocial_debt_mechanic
- integrity_test_values_reasoning
- seasonal_deal_calendar_7_seasons
- exclusivity_lockout_opportunity_cost
- kill_clause_pr_crisis_protection
- media_kit_system
- multi_niche_crossover_compatibility_matrix
- sparse_exponential_sellout_penalty
- non_linear_fatigue_cliff
- audience_trust_follower_value_coupling
- chain_of_thought_reasoning_logged
- payment_structure_expected_value
- brand_relationship_memory_rolodex
```

---

##  License

MIT - see [LICENSE](LICENSE)

---

*Built for the Meta PyTorch OpenEnv Hackathon. The environment's thesis: the creator economy is not just about money — it's about relationships, authenticity, and long-term positioning. An agent that learns this beats one that only optimizes revenue.*p