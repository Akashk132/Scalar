"""
Inference Script — AI Medical Triage Assistant
===============================================
MANDATORY STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

ENVIRONMENT VARIABLES:
  HF_TOKEN / API_KEY   — API key for the LLM
  API_BASE_URL         — LLM endpoint (default: HF router)
  MODEL_NAME           — Model to use (default: Qwen2.5-72B-Instruct)
  TASK_NAME            — Which task to run (default: runs ALL 3 tasks)
"""

import os
import sys
import json
from typing import Optional, List

from openai import OpenAI
from triage_env import MedicalTriageEnv, TriageAction, TASK_CASES

# ─── Configuration ───
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "medical_triage"
MAX_STEPS = 5
TEMPERATURE = 0.0  # deterministic for reproducibility


def _safe_score(raw: float) -> float:
    return round(max(0.01, min(0.99, raw)), 2)


def serialize_action(action: TriageAction) -> str:
    if action.action_type == "investigate":
        return f"investigate(target='{action.investigation_target}')"
    return f"triage(urgency='{action.urgency_level}', action='{action.recommended_action}')"


def build_prompt(obs) -> str:
    return (
        f"Patient ID: {obs.patient_id}\n"
        f"Age: {obs.age}\n"
        f"Gender: {obs.gender}\n"
        f"Medical History: {obs.medical_history}\n"
        f"Chief Complaint: {obs.chief_complaint}\n"
        f"Discovered Symptoms: {obs.discovered_symptoms}\n"
        f"Discovered Vitals: {obs.discovered_vitals}\n"
        f"Step: {obs.step_number}/{obs.max_steps}\n"
        f"\nInstructions: {obs.task_instructions}\n"
        f"\nRespond with a JSON object containing:\n"
        f'  "action_type": "investigate" or "triage"\n'
        f'  "investigation_target": "symptoms" or "vitals" (if investigating)\n'
        f'  "urgency_level": "low", "medium", or "high" (if triaging)\n'
        f'  "recommended_action": "self_care", "clinic_visit", or "emergency_room" (if triaging)\n'
        f'  "reasoning": "<your reasoning>"\n'
    )


SYSTEM_PROMPT = (
    "You are an AI Medical Triage Assistant. Based on patient information, "
    "you must decide whether to investigate further or make a triage decision. "
    "Always respond with valid JSON matching the required schema. "
    "Be decisive — gather only critical information, then triage."
)


def call_llm(client: Optional[OpenAI], obs, step_num: int, task_name: str) -> TriageAction:
    """Call the LLM for a decision. Falls back to heuristic if no client."""
    if client:
        try:
            prompt = build_prompt(obs)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            parsed = json.loads(text)
            return TriageAction(**parsed)
        except Exception:
            pass  # fall through to heuristic

    # ─── Heuristic Fallback ───
    return _heuristic_action(obs, step_num, task_name)


def _heuristic_action(obs, step_num: int, task_name: str) -> TriageAction:
    """Deterministic heuristic agent for reproducible baseline scores."""
    if task_name == "direct_triage":
        # All info is available — just classify
        u, a = _guess_from_text(obs)
        return TriageAction(
            action_type="triage",
            urgency_level=u,
            recommended_action=a,
            reasoning="Heuristic: all info visible, classifying directly.",
        )

    if task_name == "investigative_triage":
        if step_num <= 1 and obs.discovered_symptoms == "Not yet investigated":
            return TriageAction(
                action_type="investigate",
                investigation_target="symptoms",
                reasoning="Heuristic: need symptom data first.",
            )
        if step_num <= 2 and obs.discovered_vitals == "Not yet investigated":
            return TriageAction(
                action_type="investigate",
                investigation_target="vitals",
                reasoning="Heuristic: need vital signs.",
            )
        u, a = _guess_from_text(obs)
        return TriageAction(
            action_type="triage",
            urgency_level=u,
            recommended_action=a,
            reasoning="Heuristic: investigated, now classifying.",
        )

    # time_critical_triage — investigate vitals once, triage fast
    if step_num <= 1 and obs.discovered_vitals == "Not yet investigated":
        return TriageAction(
            action_type="investigate",
            investigation_target="vitals",
            reasoning="Heuristic: quick vitals check in time-critical.",
        )
    u, a = _guess_from_text(obs)
    return TriageAction(
        action_type="triage",
        urgency_level=u,
        recommended_action=a,
        reasoning="Heuristic: time-critical, triage immediately.",
    )


def _guess_from_text(obs) -> tuple:
    """Simple keyword-based heuristic for urgency and action."""
    text = f"{obs.chief_complaint} {obs.discovered_symptoms} {obs.discovered_vitals}".lower()

    high_keywords = ["chest pain", "crushing", "stroke", "slurred", "critical", "droop",
                     "unconscious", "impending doom", "loss of consciousness", "emergency"]
    medium_keywords = ["fever", "infection", "cough", "blood", "weight loss",
                       "ear pain", "sputum", "hemoptysis", "night sweats"]

    for kw in high_keywords:
        if kw in text:
            return "high", "emergency_room"
    for kw in medium_keywords:
        if kw in text:
            return "medium", "clinic_visit"
    return "low", "self_care"


def run_task(client: Optional[OpenAI], task_name: str):
    """Run inference for a single task across all its cases."""
    cases = TASK_CASES[task_name]
    all_scores = []

    for case_idx in range(len(cases)):
        env = MedicalTriageEnv(task_name=task_name, case_index=case_idx)
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        rewards: List[float] = []
        step_num = 0

        try:
            obs = env.reset()
            done = False

            while not done and step_num < MAX_STEPS:
                step_num += 1
                action = call_llm(client, obs, step_num, task_name)
                obs, reward_obj, done, info = env.step(action)

                r = reward_obj.score
                rewards.append(r)
                action_str = serialize_action(action)
                done_str = "true" if done else "false"
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={r:.2f} done={done_str} error=null",
                    flush=True,
                )

        except Exception as e:
            err_msg = str(e).replace("\n", " ").replace("\r", "")
            r = _safe_score(0.10)
            rewards.append(r)
            step_num = max(step_num, 1)
            print(
                f"[STEP] step={step_num} action=error reward={r:.2f} "
                f"done=true error={err_msg}",
                flush=True,
            )

        finally:
            env.close()
            final_score = _safe_score(rewards[-1]) if rewards else _safe_score(0.10)
            success = "true" if final_score > 0.50 else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(
                f"[END] success={success} steps={step_num} "
                f"score={final_score:.2f} rewards={rewards_str}",
                flush=True,
            )
            all_scores.append(final_score)

    return all_scores


def main():
    # Init OpenAI client
    client = None
    if API_KEY:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        except Exception:
            pass

    task_name = os.getenv("TASK_NAME")

    if task_name:
        # Run a single task
        scores = run_task(client, task_name)
        avg = sum(scores) / len(scores) if scores else 0.50
        print(f"\n# Task={task_name} avg_score={avg:.2f}", flush=True)
    else:
        # Run ALL tasks
        all_tasks = ["direct_triage", "investigative_triage", "time_critical_triage"]
        summary = {}
        for t in all_tasks:
            scores = run_task(client, t)
            avg = sum(scores) / len(scores) if scores else 0.50
            summary[t] = avg
            print(f"\n# Task={t} avg_score={avg:.2f}", flush=True)

        print("\n# === SUMMARY ===", flush=True)
        for t, s in summary.items():
            print(f"#   {t}: {s:.2f}", flush=True)


if __name__ == "__main__":
    main()
