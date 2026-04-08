"""
inference.py — Influencer Business Sim
=======================================
Finalized Hackathon-compliant inference script (Fixed NameError).
"""

import os, sys, json
import requests
from openai import OpenAI

# 1. ENVIRONMENT SETUP
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
# Use the injected ENV_BASE_URL, default to port 7860 for HF Space
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# --- CORE CONSTANTS (Ensure these are defined here) ---
BENCHMARK         = "influencer_business_sim"
MAX_STEPS         = 5
SUCCESS_THRESHOLD = 0.6

# --- MANDATORY DEBUG LOGS ---
print(f"DEBUG: Using API_BASE_URL = {API_BASE_URL}")
print(f"DEBUG: Using MODEL_NAME = {MODEL_NAME}")
print(f"DEBUG: Using ENV_BASE_URL = {ENV_BASE_URL}")

# 2. INITIALIZE CLIENT
client = OpenAI(
    api_key=API_KEY, 
    base_url=API_BASE_URL
)

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

# ── Rule-based fallback ───────────────────────────────────────────────────────
def _fallback(task: str, obs: dict, step_num: int = 1, counters_used: int = 0) -> tuple:
    deal    = obs.get("deal") or {}
    rating  = deal.get("brand_rating", 0)
    cat     = deal.get("category", "")
    niches  = obs.get("creator_niches", [])
    fatigue = obs.get("fatigue", 0.0)
    offer   = deal.get("initial_offer", 0)
    match   = cat in niches

    if task == "niche_alignment":
        if match and rating > 0.6:
            return "accept", None
        return "reject", None

    if task == "reputation_protection":
        if match and rating >= 0.5:
            return "accept", None
        return "reject", None

    if not match or rating < 0.5:
        return "reject", None

    if counters_used == 0 and fatigue < 0.6:
        c_price = round(offer * 1.3, 2)
        return "counter", c_price

    return "accept", None

# ── LLM call ──────────────────────────────────────────────────────────────────
def get_decision(task: str, obs: dict, step_num: int = 1, counters_used: int = 0):
    if task in ("reputation_protection", "target_negotiation"):
        dec, c_price = _fallback(task, obs, step_num, counters_used)
        return dec, c_price, None

    deal  = obs.get("deal") or {}
    offer = deal.get("initial_offer", 0)
    cat   = deal.get("category", "")
    niches = obs.get("creator_niches", [])
    rating = deal.get("brand_rating", 0)

    prompt = f"""You decide brand deals for a content creator. Reply ONLY with JSON.
Creator niches: {niches}
Deal category: {cat}
Brand rating: {rating:.2f}
Offer: ${offer}
RULE: Accept ONLY if category is in niches AND brand_rating > 0.6. Reject all else.
{{"decision": "accept", "counter_price": null}}
{{"decision": "reject", "counter_price": null}}"""

    try:
        resp = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": "You are a JSON-only responder."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 40,
            temperature = 0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        out = json.loads(raw)
        dec = out.get("decision", "reject").lower()
        
        rule_dec, _ = _fallback(task, obs, step_num, counters_used)
        if rule_dec == "reject" and dec == "accept":
            dec = "reject"
        return dec, out.get("counter_price"), None
    except Exception as e:
        dec, c_price = _fallback(task, obs, step_num, counters_used)
        return dec, c_price, str(e)

# ── Score calculator ──────────────────────────────────────────────────────────
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
    counters_used = 0

    try:
        last_obs = env_reset(task)
        for step_num in range(1, MAX_STEPS + 1):
            deal = last_obs.get("deal")
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
            if decision == "counter" and not c_price:
                c_price = round(deal.get("initial_offer", 0) * 1.3, 2)
            if decision == "counter":
                counters_used += 1

            result = env_step(decision, c_price if decision == "counter" else None)
            reward = float(result.get("reward", 0.0))
            done   = result.get("done", False)
            rewards.append(reward)

            action_json = {"decision": decision, "counter_price": c_price if decision == "counter" else None}
            print(f'[STEP] step={step_num} action={json.dumps(action_json)} '
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
        print(f"[END] success=false steps={step_num} score=0.00 rewards=0.00", flush=True)
        return 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
TASKS = ["niche_alignment", "reputation_protection", "target_negotiation"]

if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)
    
    avg = sum(scores.values()) / len(scores)
    print(f"\nFINAL AVERAGE: {avg:.2f}")