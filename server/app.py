from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from triage_env import MedicalTriageEnv, TriageAction, TriageObservation

app = FastAPI(
    title="AI Medical Triage Assistant — OpenEnv",
    description="RL environment for sequential patient triage under partial observability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
_env = MedicalTriageEnv(task_name="direct_triage")


@app.get("/")
def root():
    """Health check / HF Spaces ping endpoint — must return 200."""
    return {
        "status": "ok",
        "name": "medical_triage",
        "description": "AI Medical Triage Assistant — OpenEnv RL Environment",
        "tasks": ["direct_triage", "investigative_triage", "time_critical_triage"],
    }


@app.api_route("/reset", methods=["GET", "POST"])
def reset(task_name: str = "direct_triage", case_index: int = 0):
    """Reset the environment and return the initial observation."""
    global _env
    _env = MedicalTriageEnv(task_name=task_name, case_index=case_index)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: TriageAction):
    """Take an action and return observation, reward, done, info."""
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return current environment state."""
    return _env.state()


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
