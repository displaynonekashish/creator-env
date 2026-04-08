from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from pydantic import Field
from typing import Optional, List, Dict, Any

class Action(OpenEnvAction):
    decision: str = Field(..., description="accept / reject / counter")
    counter_price: Optional[float] = Field(None, description="Requested price for negotiation")

class Observation(OpenEnvObservation):
    creator_niches: List[str] = Field(..., description="Active niches (e.g. ['fitness', 'tech'])")
    followers: int = Field(..., description="Total reach")
    engagement_rate: float = Field(..., description="Audience quality (0.0-1.0)")
    reputation: float = Field(..., description="Brand safety score (0.0-1.0)")
    fatigue: float = Field(..., description="Burnout level (0.0-1.0)")
    earnings: float = Field(..., description="Total cash earned")
    monthly_target: float = Field(..., description="Revenue goal for 1.0 score")
    deal: Optional[Dict[str, Any]] = Field(None, description="Current deal info")
    
    # Added fields to match openenv.yaml and environment logic
    task: str = Field(..., description="Active task id")
    step: int = Field(..., description="Current step index (0-based)")
    max_steps: int = Field(..., description="Total steps per episode")