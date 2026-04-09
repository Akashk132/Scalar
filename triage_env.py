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
        "id": "pt-010", "age": 28, "gender": "Female", "history": "None",
        "symptoms": ["Mild headache and runny nose for 2 days. No fever.", "Stable."],
        "vitals": ["Temp: 37.0°C, HR: 72, BP: 110/70", "Stable."],
        "true_urgency": "low", "true_action": "self_care",
    },
    {
        "id": "pt-011", "age": 55, "gender": "Male", "history": "Type 2 Diabetes",
        "symptoms": ["Severe crushing chest pain radiating to left arm.", "Patient is losing consciousness."],
        "vitals": ["Temp: 36.5°C, HR: 110, BP: 160/95", "CRITICAL DROP"],
        "true_urgency": "high", "true_action": "emergency_room",
    },
    {
        "id": "pt-012", "age": 6, "gender": "Male", "history": "None",
        "symptoms": ["Fever of 39.5°C, lethargic, pulling at ear.", "Screaming in pain."],
        "vitals": ["Temp: 39.5°C, HR: 115", "Temp: 39.8°C"],
        "true_urgency": "medium", "true_action": "clinic_visit",
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
        self.discovered_symptoms = "Unknown"
        self.discovered_vitals = "Unknown"
        
        if self.task_name == "direct_triage":
            self.task_instructions = "Classify urgency based on upfront data."
        elif self.task_name == "investigative_triage":
            self.task_instructions = "Investigate symptoms/vitals before triage."
        else:
            self.task_instructions = "Handle deterioration during investigation."

    def reset(self) -> TriageObservation:
        self.step_number = 0
        self.penalty_accumulated = 0.0
        self.current_case = copy.deepcopy(random.choice(CASES))
        full_symptom = self.current_case["symptoms"][0]
        self.chief_complaint = full_symptom.split('.')[0] + "."
        
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
        # Maps 0.0->0.3 and 1.0->0.7 for absolute safety
        scaled = (score * 0.4) + 0.3
        return round(scaled, 4)

    def step(self, action: TriageAction) -> tuple[TriageObservation, TriageReward, bool, dict]:
        self.step_number += 1
        done = False
        true_u = self.current_case["true_urgency"]
        true_a = self.current_case["true_action"]
        
        if action.action_type == "investigate":
            if self.step_number >= self.max_steps:
                final_r = self._clamp_score(0.0)
                return self._get_observation(), TriageReward(score=final_r, reward=final_r, feedback="Timeout"), True, {"score": final_r}
            
            self.penalty_accumulated += 0.05
            if action.investigation_target == "symptoms":
                self.discovered_symptoms = self.current_case["symptoms"][0]
            else:
                self.discovered_vitals = self.current_case["vitals"][0]
            
            final_r = 0.01
            return self._get_observation(), TriageReward(score=final_r, reward=final_r, feedback="Investigating"), False, {"score": final_r}
            
        elif action.action_type == "triage":
            done = True
            u_score = 0.4 if action.urgency_level == true_u else 0.0
            a_score = 0.6 if action.recommended_action == true_a else 0.0
            
            final_score = max(0.0, (u_score + a_score) - self.penalty_accumulated)
            clamped_r = self._clamp_score(final_score)
            return self._get_observation(), TriageReward(score=clamped_r, reward=clamped_r, feedback="Triaged"), True, {"score": clamped_r, "true_urgency": true_u, "true_action": true_a}
        
        final_r = self._clamp_score(0.0)
        return self._get_observation(), TriageReward(score=final_r, reward=final_r, feedback="Err"), True, {"score": final_r}

    def close(self): pass
