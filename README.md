---
title: AI Medical Triage Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🏥 AI Medical Triage Assistant — OpenEnv RL Environment

## Real-World Motivation

In low-resource healthcare settings worldwide, patients face critical delays in receiving appropriate care. This OpenEnv simulates a **first-level AI triage assistant** that must make sequential decisions under partial observability — a genuine real-world problem.

Unlike simple classifiers, this is a **true sequential decision-making RL environment** where the agent must:
- Decide what information to gather (at a cost)
- Balance investigation thoroughness vs. time pressure
- Handle patients who deteriorate if not triaged quickly

---

## Observation Space (`TriageObservation`)

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | `str` | Unique patient identifier |
| `age` | `int` | Patient age |
| `gender` | `str` | Patient gender |
| `medical_history` | `str` | Pre-existing conditions |
| `chief_complaint` | `str` | Initial reason for visit |
| `discovered_symptoms` | `str` | Symptoms uncovered through investigation |
| `discovered_vitals` | `str` | Vital signs uncovered through investigation |
| `step_number` | `int` | Current step (1-indexed) |
| `max_steps` | `int` | Maximum steps allowed |
| `task_instructions` | `str` | Task-specific instructions |

## Action Space (`TriageAction`)

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"investigate" \| "triage"` | Choose to gather info or make final decision |
| `investigation_target` | `"symptoms" \| "vitals"` | What to investigate (if investigating) |
| `urgency_level` | `"low" \| "medium" \| "high"` | Urgency classification (if triaging) |
| `recommended_action` | `"self_care" \| "clinic_visit" \| "emergency_room"` | Care recommendation (if triaging) |
| `reasoning` | `str` | Brief justification |

## Reward Design

- **Urgency match**: up to 0.45 points (partial credit for being 1 level off)
- **Action match**: up to 0.45 points (partial credit for adjacent actions)
- **Investigation bonus**: +0.05 for thorough information gathering
- **Time penalty**: -0.06 per investigation step (-0.10 in time-critical)
- **Deterioration penalty**: -0.10 if volatile patient worsens
- **All scores clamped to strict (0.01, 0.99)** — never 0.0 or 1.0

---

## Tasks & Difficulty

### 1. `direct_triage` — Easy
Full patient information (symptoms + vitals) is visible from the start. The agent simply needs to classify urgency and recommend an action.

### 2. `investigative_triage` — Medium
Only the chief complaint is visible. The agent must spend `investigate` actions to uncover symptoms and vitals. Each investigation incurs a time penalty (-0.06). Blind guessing is penalized.

### 3. `time_critical_triage` — Hard
High-volatility patients deteriorate over time. Investigation costs more (-0.10 per step). If the agent takes too long, the patient's condition worsens, potentially changing the correct answer. The agent must triage quickly while still gathering minimum critical info.

---

## Setup & Usage

### Docker (recommended)
```bash
docker build -t openenv-triage .
docker run -p 7860:7860 openenv-triage
```
Access the API at `http://localhost:7860/`

### API Endpoints
- `GET /` — Health check (returns 200)
- `GET/POST /reset?task_name=direct_triage&case_index=0` — Reset environment
- `POST /step` — Submit an action (JSON body matching `TriageAction`)
- `GET /state` — Get current environment state

### Run Inference
```bash
export HF_TOKEN="your_api_key"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Run a specific task
```bash
export TASK_NAME="investigative_triage"
python inference.py
```

---

## Baseline Scores (Heuristic Agent)

| Task | Avg Score | Description |
|------|-----------|-------------|
| `direct_triage` | ~0.55 | Medium accuracy with keyword heuristic |
| `investigative_triage` | ~0.40 | Investigate-then-guess strategy |
| `time_critical_triage` | ~0.35 | Fast triage with partial info |

*Scores improve significantly with capable LLMs (GPT-4, Qwen-72B).*

---

## Grader Verification
```bash
python grader.py
```
All scores will be printed and verified to be in the strict (0.0, 1.0) range.
