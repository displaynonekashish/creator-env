from __future__ import annotations
"""
app.py — Influencer Business Sim v3.0
======================================
FastAPI server with full OpenEnv spec + bonus endpoints:
  POST /reset    — start new episode
  POST /step     — submit action
  GET  /state    — full internal state
  GET  /benchmark — run N episodes, return score distribution
  GET  /health
"""



import os, sys, math, random, statistics
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import (
    ResetRequest, StepRequest, StepResponse, Observation, BenchmarkResponse
)
from server.creator_env_environment import CreatorEnvEnvironment

app = FastAPI(
    title="Influencer Business Sim",
    description=(
        "OpenEnv-compliant multi-turn brand negotiation environment. "
        "An AI agent acts as a talent manager balancing Profit, Reputation, and Creator Fatigue. "
        "Features: 5 brand personalities (SHADY trap), multi-turn negotiation with mood states, "
        "gifted deal early-believer system, parasocial debt, integrity test, seasonal calendar."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_env: CreatorEnvEnvironment = CreatorEnvEnvironment()


@app.get("/")
def root():
    return {
        "status":  "ok",
        "env":     "influencer_business_sim",
        "version": "3.0.0",
        "tasks":   ["niche_alignment", "reputation_protection", "target_negotiation"],
        "features": [
            "multi_turn_negotiation_8_turns",
            "shady_brand_trap",
            "5_brand_personalities_hidden",
            "gifted_early_believer_system",
            "parasocial_debt_mechanic",
            "integrity_test",
            "seasonal_deal_calendar",
            "exclusivity_lockout",
            "kill_clause_negotiation",
            "media_kit_system",
            "multi_niche_creator_profiles",
            "sparse_exponential_sellout_penalty",
            "chain_of_thought_reasoning",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None): # Set default to None here
    global _env
    # If req is None, use default task
    task = req.task if req else "niche_alignment"
    _env = CreatorEnvEnvironment(task=task)
    return _env.reset()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        return _env.step(req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
def state():
    return _env.get_state()


@app.get("/benchmark", response_model=BenchmarkResponse)
def benchmark(
    n_episodes: int = Query(default=10, ge=1, le=50),
    task: str       = Query(default="target_negotiation"),
):
    """
    Run N full episodes with random actions and return score distribution.
    Used by judges to assess environment variance and difficulty.
    """
    scores        = []
    burnouts      = 0
    sellout_eps   = 0
    shady_caught  = 0
    integrity_pass = 0
    integrity_total = 0
    ambassador     = False

    for _ in range(n_episodes):
        env = CreatorEnvEnvironment(task=task)
        obs = env.reset()
        done = False

        for _step in range(env.MAX_STEPS):
            if done:
                break
            # Random valid action for benchmark diversity
            from models import Action, DecisionType
            actions = [
                DecisionType.ACCEPT, DecisionType.REJECT,
                DecisionType.COUNTER_20, DecisionType.COUNTER_10,
                DecisionType.POST_ORGANIC_CONTENT,
            ]
            action = Action(decision=random.choice(actions))
            resp   = env.step(action)
            obs    = resp.observation
            done   = resp.done

        final = env._calculate_final_score()
        scores.append(final)

        if obs.fatigue >= 1.0:
            burnouts += 1
        if obs.sellout_count >= 2:
            sellout_eps += 1
        if obs.shady_deals_detected > 0:
            shady_caught += 1
        if obs.integrity_test_triggered:
            integrity_total += 1
            if obs.integrity_test_passed:
                integrity_pass += 1

    mean  = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    return BenchmarkResponse(
        n_episodes=n_episodes,
        mean_score=round(mean, 3),
        std_dev=round(stdev, 3),
        best_episode=round(max(scores), 3),
        worst_episode=round(min(scores), 3),
        sellout_rate=round(sellout_eps / n_episodes, 3),
        burnout_rate=round(burnouts / n_episodes, 3),
        shady_brand_detected_rate=round(shady_caught / n_episodes, 3),
        ambassador_unlocked=ambassador,
        integrity_test_passed_rate=round(
            integrity_pass / max(1, integrity_total), 3
        ),
        per_episode_scores=scores,
    )


def main():
    mode = os.environ.get("MODE", "server")
    port = int(os.environ.get("PORT", 7860))

    if mode == "inference":
        import subprocess
        subprocess.run([sys.executable, "inference.py"], check=True)
    else:
        uvicorn.run(
            "server.app:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()