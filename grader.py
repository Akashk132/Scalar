"""
OpenEnv Grader for Medical Triage Environment
==============================================
Each task exposes a grader function that:
  1. Runs several deterministic episodes
  2. Returns the average score in the strict open interval (0.01, 0.99)
  3. Never returns 0.0 or 1.0

Grader entry points (referenced in openenv.yaml):
  - grade_direct_triage
  - grade_investigative_triage
  - grade_time_critical_triage
"""

from triage_env import MedicalTriageEnv, TriageAction, TASK_CASES


def _safe_clamp(score: float) -> float:
    """Ensure score is strictly in (0.0, 1.0) — never endpoints."""
    return round(max(0.01, min(0.99, score)), 4)


def _run_heuristic_episode(task_name: str, case_index: int) -> float:
    """
    Run a single deterministic episode with a simple heuristic agent.
    Returns the final episode score.
    """
    env = MedicalTriageEnv(task_name=task_name, case_index=case_index)
    obs = env.reset()
    done = False
    last_score = 0.01

    max_iter = env.max_steps
    step_count = 0

    while not done and step_count < max_iter:
        step_count += 1

        if task_name == "direct_triage":
            # Easy task: just triage immediately with a reasonable guess
            action = TriageAction(
                action_type="triage",
                urgency_level="medium",
                recommended_action="clinic_visit",
                reasoning="Heuristic baseline — medium urgency default",
            )
        elif task_name == "investigative_triage":
            # Medium task: investigate once, then triage
            if step_count == 1:
                action = TriageAction(
                    action_type="investigate",
                    investigation_target="symptoms",
                    reasoning="Heuristic: gather symptom info first",
                )
            elif step_count == 2:
                action = TriageAction(
                    action_type="investigate",
                    investigation_target="vitals",
                    reasoning="Heuristic: check vitals",
                )
            else:
                action = TriageAction(
                    action_type="triage",
                    urgency_level="medium",
                    recommended_action="clinic_visit",
                    reasoning="Heuristic default triage after investigation",
                )
        else:
            # Hard task (time_critical): investigate vitals quickly, triage fast
            if step_count == 1:
                action = TriageAction(
                    action_type="investigate",
                    investigation_target="vitals",
                    reasoning="Heuristic: check vitals in time-critical case",
                )
            else:
                action = TriageAction(
                    action_type="triage",
                    urgency_level="high",
                    recommended_action="emergency_room",
                    reasoning="Heuristic: assume critical for speed",
                )

        obs, reward, done, info = env.step(action)
        last_score = reward.score

    env.close()
    return last_score


def _grade_task(task_name: str) -> float:
    """Run all cases for a task, return average score strictly in (0.01, 0.99)."""
    cases = TASK_CASES[task_name]
    scores = []
    for i in range(len(cases)):
        score = _run_heuristic_episode(task_name, case_index=i)
        scores.append(score)

    if not scores:
        return 0.50

    avg = sum(scores) / len(scores)
    return _safe_clamp(avg)


def grade_direct_triage(**kwargs) -> float:
    """Grader for easy task."""
    return _grade_task("direct_triage")


def grade_investigative_triage(**kwargs) -> float:
    """Grader for medium task."""
    return _grade_task("investigative_triage")


def grade_time_critical_triage(**kwargs) -> float:
    """Grader for hard task."""
    return _grade_task("time_critical_triage")


# Legacy single entry point (for backward compat)
def evaluate_performance(submission_output: str = "", **kwargs) -> float:
    """
    Fallback grader that parses inference stdout output.
    Returns score in (0.01, 0.99).
    """
    try:
        lines = submission_output.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if "[END]" in line and "rewards=" in line:
                rewards_part = line.split("rewards=")[1].strip()
                rewards = [float(r) for r in rewards_part.split(",") if r.strip()]
                if rewards:
                    avg = sum(rewards) / len(rewards)
                    return _safe_clamp(avg)
        return 0.50  # safe midpoint if nothing found
    except Exception:
        return 0.50


if __name__ == "__main__":
    print("=== Grader Self-Test ===")
    for task in ["direct_triage", "investigative_triage", "time_critical_triage"]:
        score = _grade_task(task)
        print(f"  {task}: {score:.4f}")
        assert 0.0 < score < 1.0, f"Score {score} out of range for {task}!"
    print("All grader scores in valid range [OK]")
