from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from triage_env import MedicalTriageEnv, TriageAction

app = FastAPI(title="AI Medical Triage Assistant Env & UI")

# Allow CORS for potential local UI dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Env for the Hackathon validation & UI demo
global_env = MedicalTriageEnv(task_name="direct_triage")

# ---- API Endpoints ----
@app.get("/api/ping")
def ping():
    return {"status": "ok", "message": "Medical Triage Environment API is running."}

@app.api_route("/reset", methods=["GET", "POST"])
def reset(task_name: str = "investigative_triage"):
    global global_env
    global_env = MedicalTriageEnv(task_name=task_name)
    obs = global_env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: TriageAction):
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return global_env.state()


# ---- Web UI UI ----
# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Returns the beautiful Glassmorphism Web UI and serves as a 200 OK for HF Spaces Ping"""
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI Not Found</h1><p>But API is running!</p>"

def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))

if __name__ == "__main__":
    main()
