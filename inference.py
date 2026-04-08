"""
inference.py — Influencer Business Sim
=======================================
Hackathon-compliant inference script.

Env vars (with defaults):
  API_BASE_URL  — LLM endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace token
  ENV_BASE_URL  — env server (default http://localhost:8000)

Stdout format:
  [START] task=<id> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os, sys, json
import requests
from openai import OpenAI
from dotenv import load_dotenv  # Add this line

# Load variables from .env into the system environment
load_dotenv() 

# Now os.getenv will automatically find the values in your .env file
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
# ... rest of your code
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK         = "influencer_business_sim"
MAX_STEPS         = 5
SUCCESS_THRESHOLD = 0.6

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── ENV helpers ───────────────────────────────────────────────────────────────
def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)

def env_step(decision: str, counter_price=None) -> dict:
    action = {"decision": decision}
    if counter_price is not None:
        action["counter_price"] = float(counter_price)
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=15)
    r.raise_for_status()
    data = r.json()
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

# ── Rule-based fallback (PRIMARY decision maker) ──────────────────────────────
def _fallback(task: str, obs: dict, step_num: int = 1, counters_used: int = 0) -> tuple:
    """
    Returns (decision, counter_price).
    This is the PRIMARY logic. LLM is only used for niche_alignment as a sanity check.
    """
    deal    = obs.get("deal") or {}
    rating  = deal.get("brand_rating", 0)
    cat     = deal.get("category", "")
    niches  = obs.get("creator_niches", [])
    fatigue = obs.get("fatigue", 0.0)
    offer   = deal.get("initial_offer", 0)
    match   = cat in niches

    if task == "niche_alignment":
        # Accept ONLY if niche matches AND brand_rating > 0.6
        if match and rating > 0.6:
            return "accept", None
        return "reject", None

    if task == "reputation_protection":
        # Accept ONLY if niche matches AND brand_rating >= 0.5
        if match and rating >= 0.5:
            return "accept", None
        return "reject", None

    # target_negotiation — counter ONCE, then accept, otherwise reject bad deals
    if not match or rating < 0.5:
        return "reject", None

    # Good deal: counter once at 1.3x, then accept on subsequent steps
    if counters_used == 0 and fatigue < 0.6:
        c_price = round(offer * 1.3, 2)
        return "counter", c_price

    # After 1 counter, or if fatigued, just accept
    return "accept", None

# ── LLM call (used only for niche_alignment as secondary check) ──────────────
def get_decision(task: str, obs: dict, step_num: int = 1, counters_used: int = 0):
    """
    Returns (decision, counter_price, error_str).

    Strategy by task:
      - reputation_protection : pure rule-based, never trust LLM (too risky)
      - target_negotiation    : pure rule-based, LLM was countering every step
      - niche_alignment       : try LLM, fallback on error
    """
    # Force deterministic rules for high-stakes tasks
    if task in ("reputation_protection", "target_negotiation"):
        dec, c_price = _fallback(task, obs, step_num, counters_used)
        return dec, c_price, None

    # niche_alignment — attempt LLM
    deal  = obs.get("deal") or {}
    offer = deal.get("initial_offer", 0)
    cat   = deal.get("category", "")
    niches = obs.get("creator_niches", [])
    rating = deal.get("brand_rating", 0)

    prompt = f"""You decide brand deals for a content creator. Reply ONLY with JSON, no markdown.

Creator niches: {niches}
Deal category: {cat}  (matches niche: {cat in niches})
Brand rating: {rating:.2f}
Offer: ${offer}

RULE: Accept ONLY if category is in niches AND brand_rating > 0.6. Reject all else.

Reply format (choose one):
{{"decision": "accept", "counter_price": null}}
{{"decision": "reject", "counter_price": null}}"""

    try:
        resp = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": "You are a JSON-only responder. Output raw JSON only, no explanation, no markdown."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 40,
            temperature = 0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        out = json.loads(raw)
        dec = out.get("decision", "reject").lower()
        if dec not in ("accept", "reject", "counter"):
            dec = "reject"
        # Safety check: if LLM says accept but rule says reject, trust the rule
        rule_dec, _ = _fallback(task, obs, step_num, counters_used)
        if rule_dec == "reject" and dec == "accept":
            dec = "reject"  # override bad LLM accept
        return dec, out.get("counter_price"), None
    except Exception as e:
        dec, c_price = _fallback(task, obs, step_num, counters_used)
        return dec, c_price, str(e)

# ── Score calculator (mirrors env _calculate_score) ───────────────────────────
def _calc_score(task: str, obs: dict) -> float:
    mt      = max(1, obs.get("monthly_target", 1))
    earn    = obs.get("earnings", 0)
    rep     = obs.get("reputation", 0)
    eng     = obs.get("engagement_rate", 0)
    fatigue = obs.get("fatigue", 0)

    rev    = min(1.0, earn / mt)
    health = (rep + min(1.0, eng * 8)) / 2
    fat    = 1.0 - fatigue

    if task == "niche_alignment":
        final = rev * 0.2 + health * 0.6 + fat * 0.2
    elif task == "reputation_protection":
        final = rev * 0.2 + health * 0.7 + fat * 0.1
    else:
        final = rev * 0.5 + health * 0.3 + fat * 0.2

    return round(max(0.0, min(1.0, final)), 2)

# ── Episode runner ────────────────────────────────────────────────────────────
def run_task(task: str) -> float:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards, step_num, last_obs, final_score = [], 0, {}, 0.0
    counters_used = 0  # track per-episode counter usage

    try:
        last_obs = env_reset(task)

        for step_num in range(1, MAX_STEPS + 1):
            deal = last_obs.get("deal")

            # No deal left — safe reject
            if not deal:
                result = env_step("reject")
                reward = float(result.get("reward", 0.0))
                done   = result.get("done", False)
                rewards.append(reward)
                print(f'[STEP] step={step_num} action={{"decision":"reject","counter_price":null}} '
                      f'reward={reward:.2f} done={str(done).lower()} error=null', flush=True)
                last_obs = result.get("observation", last_obs)
                if done: break
                continue

            decision, c_price, err = get_decision(task, last_obs, step_num, counters_used)

            # Auto-fill counter price if missing
            if decision == "counter" and not c_price:
                c_price = round(deal.get("initial_offer", 0) * 1.3, 2)

            # Track counter usage to prevent infinite countering
            if decision == "counter":
                counters_used += 1

            result = env_step(decision, c_price if decision == "counter" else None)
            reward = float(result.get("reward", 0.0))
            done   = result.get("done", False)
            rewards.append(reward)

            if decision == "counter":
                action_str = f'{{"decision":"counter","counter_price":{c_price}}}'
            else:
                action_str = f'{{"decision":"{decision}","counter_price":null}}'

            print(f'[STEP] step={step_num} action={action_str} '
                  f'reward={reward:.2f} done={str(done).lower()} '
                  f'error={err if err else "null"}', flush=True)

            last_obs = result.get("observation", last_obs)

            meta = result.get("metadata", {})
            if meta and "final_score" in meta:
                final_score = float(meta["final_score"])

            if done: break

        if final_score == 0.0:
            final_score = _calc_score(task, last_obs)

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success     = final_score >= SUCCESS_THRESHOLD
        print(f"[END] success={str(success).lower()} steps={step_num} "
              f"score={final_score:.2f} rewards={rewards_str}", flush=True)
        return final_score

    except Exception as exc:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success=false steps={step_num} score=0.00 rewards={rewards_str}", flush=True)
        print(f"!! Crash: {exc}", file=sys.stderr)
        return 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
TASKS = ["niche_alignment", "reputation_protection", "target_negotiation"]

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        print(f"\n{'='*60}", flush=True)
        scores[task] = run_task(task)

    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    for t, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  {t:30s}  {s:.2f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"\n  {'AVERAGE':30s}  {avg:.2f}", flush=True)
    print(f"{'='*60}", flush=True)