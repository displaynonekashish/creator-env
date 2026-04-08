# 🚀 Influencer Business Sim: A Real-World Agent Environment

The **Influencer Business Sim** is a high-fidelity OpenEnv environment designed to evaluate AI agents on their ability to act as professional talent managers. Unlike toy environments, this simulation models the multi-dimensional trade-offs of the creator economy.

## 🌟 Real-World Utility
Content creators face a daily barrage of brand offers. An effective agent must maximize revenue without destroying the creator's brand safety or causing mental burnout. This environment bridges the gap between pure RL and practical business automation.

## 🛠 Architecture & Spec Compliance
- **Full OpenEnv Spec:** Implementation of `step()`, `reset()`, and `state()` via a containerized FastAPI server.
- **Pydantic Driven:** Strictly typed `Observation` and `Action` models for robust agent-environment communication.
- **Dockerized:** Ready for multi-mode deployment on Hugging Face Spaces.

## 📊 Environment Mechanics
- **Action Space:** - `accept`: Signs the deal.
  - `reject`: Passes on the deal.
  - `counter`: Negotiates for a higher price (increases reward but adds fatigue).
- **State Space:** Tracks 11 unique variables including reputation (0-1), fatigue (0-1), and audience engagement.
- **Reward Shaping:** Partial progress signals are provided to penalize "greedy" or "destructive" behavior while rewarding strategic negotiation.

## 🎯 Task Suite
| Task | Difficulty | Objective |
| :--- | :--- | :--- |
| **Niche Alignment** | Easy | Filter deals based on category relevance. |
| **Reputation Protection** | Medium | Stricter brand-safety gatekeeping. |
| **Target Negotiation** | Hard | Hit $6,250 revenue target via counters while managing fatigue. |

## 📈 Baseline Scores
| Task | Score |
| :--- | :--- |
| **Average Score** | **0.73** |
| Success Threshold | 0.60 |

*Evaluated using Meta-Llama-3-8B-Instruct.*

## 🚀 Setup & Execution
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Start Environment:** `python server/app.py`
3. **Run Inference:** `python inference.py`