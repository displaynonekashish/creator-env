import random
import asyncio
from openenv.core.env_server.interfaces import Environment
try:
    from .models import Observation, Action
except ImportError:
    from models import Observation, Action

class CreatorEnvironment(Environment):
    def __init__(self):
        self.max_steps      = 5
        self.step_count     = 0
        self.total_earnings = 0.0
        self.reputation     = 0.8
        self.fatigue        = 0.0
        self._shared_deals  = []
        self._creator_niches = ["tech", "fitness"]
        self._followers      = 250000
        self._engagement_rate = 0.06
        self._monthly_target  = 6250.0
        self._current_task   = "niche_alignment"

    async def reset(self, request=None):
        self.step_count     = 0
        self.total_earnings = 0.0
        self.reputation     = 0.8
        self.fatigue        = 0.0
        
        if request:
            if hasattr(request, "task") and request.task:
                self._current_task = request.task
            elif isinstance(request, dict) and request.get("task"):
                self._current_task = request["task"]

        all_cats = ["fitness", "tech", "skincare"]
        self._shared_deals = []
        for i in range(self.max_steps):
            cat = random.choice(self._creator_niches) if i < 3 else random.choice(all_cats)
            rating = round(random.uniform(0.4, 0.9), 2)
            self._shared_deals.append({
                "brand": f"Brand_{random.randint(10, 99)}",
                "initial_offer": random.randint(1000, 4000),
                "category": cat,
                "brand_rating": rating,
            })
        return self._get_obs()

    def state(self):
        return self._get_obs()

    def _get_obs(self):
        deal_info = self._shared_deals[self.step_count] if self.step_count < len(self._shared_deals) else None
        return Observation(
            creator_niches  = self._creator_niches,
            followers       = self._followers,
            engagement_rate = float(self._engagement_rate),
            reputation      = float(self.reputation),
            fatigue         = float(self.fatigue),
            earnings        = float(self.total_earnings),
            monthly_target  = float(self._monthly_target),
            deal            = deal_info,
            task            = self._current_task,
            step            = self.step_count,
            max_steps       = self.max_steps
        )

    def _calculate_score(self):
        rev_score   = min(1.0, self.total_earnings / max(1, self._monthly_target))
        health_score = (self.reputation + min(1.0, self._engagement_rate * 8)) / 2
        fatigue_score = 1.0 - self.fatigue

        if self._current_task == "niche_alignment":
            final = (rev_score * 0.2) + (health_score * 0.6) + (fatigue_score * 0.2)
        elif self._current_task == "reputation_protection":
            final = (rev_score * 0.2) + (health_score * 0.7) + (fatigue_score * 0.1)
        else: # target_negotiation
            final = (rev_score * 0.5) + (health_score * 0.3) + (fatigue_score * 0.2)

        return round(max(0.0, min(1.0, final)), 2)

    async def step(self, action: Action):
        if self.step_count >= len(self._shared_deals):
            return {"observation": self._get_obs(), "reward": 0.0, "done": True}

        current_deal = self._shared_deals[self.step_count]
        is_match = current_deal["category"] in self._creator_niches
        brand_trust = current_deal["brand_rating"]
        reward = 0.0

        if action.decision == "accept":
            if is_match and brand_trust >= 0.5:
                reward += 25.0 
                self.total_earnings += float(current_deal["initial_offer"])
                self.fatigue = min(1.0, self.fatigue + 0.2)
                self.reputation = min(1.0, self.reputation + 0.05)
            else:
                reward -= 15.0
                self.reputation -= 0.2
            
        elif action.decision == "reject":
            if not is_match or brand_trust < 0.4:
                reward += 10.0 
            else:
                reward -= 5.0 

        elif action.decision == "counter":
            if is_match and brand_trust >= 0.5 and self.fatigue < 0.6:
                reward += 12.0 
                self.fatigue = min(1.0, self.fatigue + 0.1)
            else:
                reward -= 10.0

        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return {
            "observation": self._get_obs(),
            "reward": float(reward),
            "done": done,
            "metadata": {"final_score": self._calculate_score()} if done else {}
        }