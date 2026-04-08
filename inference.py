import os
import json
from openai import OpenAI
from triage_env import MedicalTriageEnv, TriageAction

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
API_KEY = HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
TASK_NAME = os.getenv("MEDICAL_TRIAGE_TASK", "classify_urgency") # Fallback to first task
BENCHMARK = "medical_triage"

def serialize_action(action: TriageAction) -> str:
    """Format action string for stdout logs."""
    if action.action_type == "investigate":
        return f"Triage(type='investigate', target='{action.investigation_target}')"
    return f"Triage(type='triage', urgency='{action.urgency_level}', action='{action.recommended_action}')"

def main():
    # Attempt to use real client, fall back gracefully if just testing structure
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        client = None

    tasks_to_run = ["direct_triage", "investigative_triage", "time_critical_triage"]
    if os.getenv("MEDICAL_TRIAGE_TASK"):
        tasks_to_run = [os.getenv("MEDICAL_TRIAGE_TASK")]

    for task_name in tasks_to_run:
        env = MedicalTriageEnv(task_name=task_name)
        step_num = 0
        rewards = []
        success = "false"
        
        # 1. Print START line
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
        
        try:
            obs = env.reset()
            done = False
            
            action_schema = {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["investigate", "triage"]},
                    "investigation_target": {"type": "string", "enum": ["symptoms", "vitals"]},
                    "urgency_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    "recommended_action": {"type": "string", "enum": ["self_care", "clinic_visit", "emergency_room"]},
                    "reasoning": {"type": "string"}
                },
                "required": ["action_type", "reasoning"]
            }

            while not done and step_num < env.max_steps:
                step_num += 1
                error_msg = "null" # Reset per step
                
                prompt = f"Patient ID: {obs.patient_id}\nAge: {obs.age}\nGender: {obs.gender}\nHistory: {obs.medical_history}\nChief Complaint: {obs.chief_complaint}\nDiscovered Symptoms: {obs.discovered_symptoms}\nDiscovered Vitals: {obs.discovered_vitals}\n\nTask: {obs.task_instructions}\n"
                
                # Use LLM to decide action
                try:
                    if client:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a first-level AI Medical Triage Assistant operating in a low-resource setting."},
                                {"role": "user", "content": prompt}
                            ],
                            functions=[{"name": "triage", "description": "Output triage decision", "parameters": action_schema}],
                            function_call={"name": "triage"},
                            temperature=0.0
                        )
                        func_args = response.choices[0].message.function_call.arguments
                        parsed_args = json.loads(func_args)
                        action = TriageAction(**parsed_args)
                    else:
                        # Mock action for local testing without key
                        if task_name in ["investigative_triage", "time_critical_triage"] and obs.discovered_vitals == "Unknown":
                            action = TriageAction(action_type="investigate", investigation_target="vitals", reasoning="mock fallback investigation")
                        else:
                            action = TriageAction(action_type="triage", urgency_level="medium", recommended_action="clinic_visit", reasoning="mock local fallback")
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ').replace('\r', '')
                    # Fallback for API failure
                    if task_name in ["investigative_triage", "time_critical_triage"] and obs.discovered_vitals == "Unknown":
                        action = TriageAction(action_type="investigate", investigation_target="vitals", reasoning="Need to check vitals first")
                    else:
                        action = TriageAction(action_type="triage", urgency_level="medium", recommended_action="clinic_visit", reasoning="api fallback")

                # 2. Step the env
                obs, reward_obj, done, info = env.step(action)
                
                r = reward_obj.score
                rewards.append(r)
                
                # 3. Print STEP line
                action_str = serialize_action(action)
                done_str = "true" if done else "false"
                print(f"[STEP] step={step_num} action={action_str} reward={r:.2f} done={done_str} error={error_msg}")

            if any(r > 0.5 for r in rewards): success = "true"

        except Exception as e:
            # Handle catastrophic task-level failure
            pass
        finally:
            env.close()
            # 4. Print END line
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success} steps={step_num} rewards={rewards_str}")

if __name__ == "__main__":
    # If run in validation mode without args, we should run all 3 tasks to ensure it reproduces scores?
    # The hackathon validators usually pass the task via env var or loop it. 
    # Let's support running the primary loop.
    main()
