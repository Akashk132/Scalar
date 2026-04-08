---
title: AI Medical Triage Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# AI Medical Triage Assistant (OpenEnv) - RL Sequential Investigation Edition

## Real-world Motivation & Social Impact 
In many low-resource rural area settings in India, there is a severe shortage of doctors. Patients often travel long distances only to find out they could have managed their symptoms at home, or conversely, underestimate critical conditions until it is too late.

This OpenEnv simulates a first-level AI triage assistant. However, it is not a basic text classifier. It functions as a true **Sequential Decision-Making RL Environment**. The AI agent must explore partial observability, budget its actions to investigate symptoms, and manage the risk of time deterioration.

## Action & Observation Spaces
### Observation (Pydantic: `TriageObservation`)
- **`patient_id`**, **`age`** & **`gender`**: Basic demographics.
- **`medical_history`**: Pre-existing conditions.
- **`chief_complaint`**: The initial vague reason the patient walked in.
- **`discovered_symptoms`**: Blank at first. Must be uncovered by the agent.
- **`discovered_vitals`**: Blank at first. Must be uncovered by the agent.
- **`step_number`**: Current interaction step.
- **`task_instructions`**: Guide for the LLM on exactly what is required for the current task tier.

### Action (Pydantic: `TriageAction`)
- **`action_type`**: Literal `["investigate", "triage"]`.
- **`investigation_target`**: `["symptoms", "vitals"]` (used if investigating).
- **`urgency_level`**: Literal `["low", "medium", "high"]` (used if triaging).
- **`recommended_action`**: Literal `["self_care", "clinic_visit", "emergency_room"]` (used if triaging).
- **`reasoning`**: Brief justification.

## Tasks and Difficulty
1. **`direct_triage` (Easy)**: The agent has full observability immediately. It must simply map the problem state to the urgency directly.
2. **`investigative_triage` (Medium)**: The agent is presented with a vague chief complaint. It must spend `investigate` actions to retrieve vitals and symptoms. Each action adds a penalty (`-0.05`) simulating time passing. Guessing blindly gets heavily penalized.
3. **`time_critical_triage` (Hard)**: The agent operates under strict time constraints. Taking too many steps causes "Volatile" patients to deteriorate physically (dropping vitals, worsening true-urgency). The AI must identify high-risk patients efficiently and escalate quickly.

## Setup Instructions
```bash
# Clone repo and start container
docker build -t openenv-triage .
docker run -p 7860:7860 openenv-triage
```
Access the environment API locally via `http://localhost:7860/`.

## Baseline Inference
Run the baseline inference script providing an OpenAI compatible key:
```bash
export API_KEY="your_api_key_here"
export MODEL_NAME="gpt-4"
export MEDICAL_TRIAGE_TASK="investigative_triage"
python inference.py
```

Expected Baseline Scores (GPT-4-turbo mock test):
- `direct_triage`: ~ 1.00
- `investigative_triage`: ~ 0.90
- `time_critical_triage`: ~ 0.85
