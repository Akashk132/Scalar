import random
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
import copy

# --- OpenEnv Pydantic Models ---
class TriageObservation(BaseModel):
    patient_id: str = Field(description="Unique identifier for the patient")
    age: int = Field(description="Age of the patient")
    gender: str = Field(description="Gender of the patient")
    medical_history: str = Field(description="Relevant medical history")
    chief_complaint: str = Field(description="The primary reason the patient is seeking care")
    discovered_symptoms: str = Field(description="Symptoms uncovered through investigation")
    discovered_vitals: str = Field(description="Vital signs uncovered through investigation")
    step_number: int = Field(description="The current step in the interaction")
    task_instructions: str = Field(description="Specific instructions for the assistant based on the current task")

class TriageAction(BaseModel):
    action_type: Literal["investigate", "triage"] = Field(
        description="Choose 'investigate' to uncover more information. Choose 'triage' to make a final decision."
    )
    investigation_target: Optional[Literal["symptoms", "vitals"]] = Field(
        default=None, description="If action_type is 'investigate', specify what to check."
    )
    urgency_level: Optional[Literal["low", "medium", "high"]] = Field(
        default=None, description="If action_type is 'triage', provide the urgency level."
    )
    recommended_action: Optional[Literal["self_care", "clinic_visit", "emergency_room"]] = Field(
        default=None, description="If action_type is 'triage', recommend the action."
    )
    reasoning: str = Field(description="Brief reasoning for the chosen action.")

class TriageReward(BaseModel):
    score: float = Field(description="A score between 0.0 and 1.0 representing the quality of the action.")
    reward: float = Field(description="Alias for score, between 0.0 and 1.0.")
    feedback: str = Field(description="Textual feedback explaining the score.")

# --- Complex Case Datasets ---
CASES = [
    {
        "id": "pt-010",
        "age": 28, "gender": "Female", "history": "None",
        "symptoms": ["Mild headache and runny nose for 2 days. No fever.", "Symptoms are stable, just feeling a bit tired."],
        "vitals": ["Temp: 37.0°C, HR: 72, BP: 110/70", "Temp: 37.1°C, HR: 70, BP: 110/68"],
        "true_urgency": "low", "true_action": "self_care",
    },
    {
        "id": "pt-011",
        "age": 55, "gender": "Male", "history": "Type 2 Diabetes, Hypertension",
        "symptoms": ["Sudden onset of severe crushing chest pain radiating to left arm. Sweating profusely.", "Patient is losing consciousness."],
        "vitals": ["Temp: 36.5°C, HR: 110, BP: 160/95", "Temp: 36.0°C, HR: 140, BP: 90/50 - CRITICAL DROP"],
        "true_urgency": "high", "true_action": "emergency_room",
    },
    {
        "id": "pt-012",
        "age": 6, "gender": "Male", "history": "None",
        "symptoms": ["Fever of 39.5°C, lethargic, pulling at ear.", "Screaming in pain from ear, fever persists."],
        "vitals": ["Temp: 39.5°C, HR: 115", "Temp: 39.8°C, HR: 125"],
        "true_urgency": "medium", "true_action": "clinic_visit",
    },
    {
        "id": "pt-013",
        "age": 72, "gender": "Female", "history": "COPD",
        "symptoms": ["Increased shortness of breath over 2 days. Coughing more than usual.", "Gasping for air, lips turning slightly blue."],
        "vitals": ["Temp: 37.2°C, HR: 95, SpO2: 92%", "Temp: 37.4°C, HR: 115, SpO2: 84% - DANGEROUSLY LOW"],
        "true_urgency": "high", "true_action": "emergency_room",
    },
    {
        "id": "pt-014",
        "age": 22, "gender": "Female", "history": "None",
        "symptoms": ["Lower right abdominal pain, started dull now sharp. Nausea.", "Pain is severe when pressed, actively vomiting."],
        "vitals": ["Temp: 38.0°C, HR: 90, BP: 120/80", "Temp: 38.5°C, HR: 105, BP: 115/75"],
        "true_urgency": "high", "true_action": "emergency_room",
    },
    {
        "id": "pt-015",
        "age": 45, "gender": "Male", "history": "None",
        "symptoms": ["Sprained ankle during running. Mild swelling.", "Pain is manageable with ice, no worsening."],
        "vitals": ["Temp: 36.8°C, HR: 75, BP: 120/80", "Temp: 36.8°C, HR: 72, BP: 118/78"],
        "true_urgency": "low", "true_action": "self_care",
    }
]

# --- Environment Implementation ---
class MedicalTriageEnv:
    def __init__(self, task_name: str = "direct_triage"):
        self.task_name = task_name
        self.current_case = None
        self.step_number = 0
        self.max_steps = 4
        self.penalty_accumulated = 0.0
        
        self.discovered_symptoms = "None"
        self.discovered_vitals = "None"
        self.chief_complaint = ""
        
        if self.task_name == "direct_triage":
            self.task_instructions = "All info is provided upfront. Output action_type='triage' with your assessment."
        elif self.task_name == "investigative_triage":
            self.task_instructions = "You only have the chief complaint. Output action_type='investigate' with target 'symptoms' or 'vitals' to learn more. When ready, output 'triage'."
        elif self.task_name == "time_critical_triage":
            self.task_instructions = "Time constraints apply. Investigate wisely. Excess steps will cause patient deterioration. Triage as 'high' emergency if vitals crash."
        else:
            self.task_instructions = "Unknown task mode. Defaulting to direct triage."
            self.task_name = "direct_triage"

    def reset(self) -> TriageObservation:
        self.step_number = 0
        self.penalty_accumulated = 0.0
        self.current_case = copy.deepcopy(random.choice(CASES))
        full_symptom = self.current_case["symptoms"][0]
        self.chief_complaint = full_symptom.split('.')[0] + "." if '.' in full_symptom else full_symptom
        
        if self.task_name == "direct_triage":
            self.discovered_symptoms = full_symptom
            self.discovered_vitals = self.current_case["vitals"][0]
        else:
            self.discovered_symptoms = "Unknown"
            self.discovered_vitals = "Unknown"
            
        return self._get_observation()

    def _get_observation(self) -> TriageObservation:
        return TriageObservation(
            patient_id=self.current_case["id"],
            age=self.current_case["age"],
            gender=self.current_case["gender"],
            medical_history=self.current_case["history"],
            chief_complaint=self.chief_complaint,
            discovered_symptoms=self.discovered_symptoms,
            discovered_vitals=self.discovered_vitals,
            step_number=self.step_number + 1,
            task_instructions=self.task_instructions
        )

    def _clamp_score(self, score: float) -> float:
        """
        Guarantee that total task score is strictly between 0 and 1.
        Maps 0.0 -> 0.25 and 1.0 -> 0.75 for absolute safety.
        """
        scaled = (score * 0.5) + 0.25
        return round(scaled, 4)

    def step(self, action: TriageAction) -> tuple[TriageObservation, TriageReward, bool, dict]:
        self.step_number += 1
        done = False
        feedback = ""
        
        true_u = self.current_case["true_urgency"]
        true_a = self.current_case["true_action"]
        
        if action.action_type == "investigate":
            if self.step_number >= self.max_steps:
                done = True
                final_r = self._clamp_score(0.0)
                return self._get_observation(), TriageReward(score=final_r, reward=final_r, feedback="Max steps reached."), done, {"reward": final_r, "score": final_r}
                
            self.penalty_accumulated += 0.05
            
            if action.investigation_target == "symptoms":
                self.discovered_symptoms = self.current_case["symptoms"][0]
                feedback = "Symptoms discovered."
            elif action.investigation_target == "vitals":
                self.discovered_vitals = self.current_case["vitals"][0]
                feedback = "Vitals recorded."
            else:
                feedback = "Invalid investigation target."
                
            if self.task_name == "time_critical_triage" and self.step_number >= 2:
                if len(self.current_case["symptoms"]) > 1:
                    self.discovered_symptoms += " UPDATE: " + self.current_case["symptoms"][1]
                    self.discovered_vitals = self.current_case["vitals"][1]
                    true_u = "high" 
                    true_a = "emergency_room"
                    self.current_case["true_urgency"] = true_u
                    self.current_case["true_action"] = true_a
                    feedback += " PATIENT DETERIORATED!"
            
            final_r = 0.01 
            return self._get_observation(), TriageReward(score=final_r, reward=final_r, feedback=feedback), False, {"reward": final_r, "score": final_r}
            
        elif action.action_type == "triage":
            done = True
            pred_u = action.urgency_level
            pred_a = action.recommended_action
            
            u_score = 0.4 if pred_u == true_u else 0.0
            a_score = 0.6 if pred_a == true_a else 0.0
            
            if pred_u != true_u and (true_u == 'high' and pred_u == 'medium'): u_score = 0.2
            if pred_a != true_a and (true_a == 'self_care' and pred_a == 'clinic_visit'): a_score = 0.3
            
            if true_a == "emergency_room" and pred_a == "self_care":
                a_score, u_score = 0.0, 0.0
                feedback = "FATAL: Recommended self-care for emergency."
            elif pred_a == true_a and pred_u == true_u:
                feedback = "Perfect assessment."
            else:
                feedback = f"Expected [{true_u}/{true_a}]. Recommended [{pred_u}/{pred_a}]."
                
            base_score = u_score + a_score
            final_score = max(0.0, base_score - self.penalty_accumulated)
            
            if self.task_name in ["investigative_triage", "time_critical_triage"]:
                if self.discovered_vitals == "Unknown" and self.discovered_symptoms == "Unknown":
                    final_score = 0.0
                    feedback = "FATAL: Triaged blindly!"

            clamped_r = self._clamp_score(final_score)
            return self._get_observation(), TriageReward(score=clamped_r, reward=clamped_r, feedback=feedback), done, {"true_urgency": true_u, "true_action": true_a, "reward": clamped_r, "score": clamped_r}
            
        else:
            final_r = self._clamp_score(0.0)
            return self._get_observation(), TriageReward(score=final_r, reward=final_r, feedback="Invalid action"), True, {"reward": final_r, "score": final_r}

    def close(self):
        pass

    def state(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "step_number": self.step_number,
            "patient_id": self.current_case["id"] if self.current_case else None,
            "max_steps": self.max_steps,
            "penalty_accumulated": self.penalty_accumulated
        }

def evaluate_triage_performance(submission_output: str) -> float:
    """Explicit top-level grader function for the OpenEnv validator."""
    try:
        lines = submission_output.split('\n')
        for line in reversed(lines):
            if '[END]' in line and 'rewards=' in line:
                rewards_part = line.split('rewards=')[1]
                rewards = [float(r) for r in rewards_part.split(',') if r]
                if rewards:
                    # Return the average across steps, clamped to [0.25, 0.75]
                    avg = sum(rewards) / len(rewards)
                    return max(0.25, min(0.75, avg))
        return 0.25
    except Exception:
        return 0.25
