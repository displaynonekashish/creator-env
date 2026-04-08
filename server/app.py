import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.creator_env_environment import CreatorEnvironment
from models import Action 

app = FastAPI(title="Influencer Business Sim")
_env = CreatorEnvironment()

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: Action

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset(request: ResetRequest = None):
    obs = await _env.reset(request) 
    return {
        "observation": obs.model_dump(), 
        "reward": 0.0, 
        "done": False
    }

@app.post("/step")
async def step(request: StepRequest):
    result = await _env.step(request.action)
    return {
        "observation": result["observation"].model_dump(),
        "reward": result["reward"],
        "done": result["done"],
        "metadata": result.get("metadata", {})
    }

@app.get("/state")
async def state():
    return {
        "earnings": _env.total_earnings,
        "reputation": _env.reputation,
        "fatigue": _env.fatigue,
        "score": _env._calculate_score()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

