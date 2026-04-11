"""
Medical Triage RL Environment — OpenEnv Compliant
===================================================
Simulates a first-level AI medical triage assistant for low-resource settings.
The agent must investigate patients under partial observability and classify
urgency + recommend an action. Rewards are continuous in the strict open
interval (0.0, 1.0) — never exactly 0 or 1.

Tasks:
  direct_triage        (easy)   — full info upfront, just classify
  investigative_triage (medium) — must investigate before classifying
  time_critical_triage (hard)   — patient deteriorates over time
"""

import random
import copy
from typing import Dict, List, Optional, Any, Literal, Tuple
from pydantic import BaseModel, Field


# ─────────────────────── Pydantic Models ───────────────────────

class TriageObservation(BaseModel):
    """What the agent sees at each step."""
    patient_id: str = Field(description="Unique patient identifier")
    age: int = Field(description="Patient age")
    gender: str = Field(description="Patient gender")
    medical_history: str = Field(description="Pre-existing conditions")
    chief_complaint: str = Field(description="Initial reason for visit")
    discovered_symptoms: str = Field(description="Symptoms uncovered so far")
    discovered_vitals: str = Field(description="Vitals uncovered so far")
    step_number: int = Field(description="Current step (1-indexed)")
    max_steps: int = Field(description="Maximum steps allowed")
    task_instructions: str = Field(description="What the agent should do")


class TriageAction(BaseModel):
    """What the agent can do."""
    action_type: Literal["investigate", "triage"] = Field(
        description="'investigate' to gather info; 'triage' to make final decision"
    )
    investigation_target: Optional[Literal["symptoms", "vitals"]] = Field(
        default=None,
        description="What to investigate (only when action_type='investigate')"
    )
    urgency_level: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Urgency classification (only when action_type='triage')"
    )
    recommended_action: Optional[Literal["self_care", "clinic_visit", "emergency_room"]] = Field(
        default=None,
        description="Recommended care action (only when action_type='triage')"
    )
    reasoning: str = Field(default="", description="Brief justification for the action")


class TriageReward(BaseModel):
    """Reward signal returned after each step."""
    score: float = Field(description="Score in the open interval (0.0, 1.0)")
    reward: float = Field(description="Alias for score, in (0.0, 1.0)")
    feedback: str = Field(description="Human-readable feedback")


# ─────────────────────── Case Database ───────────────────────
# Each case has layered symptom/vital discovery and a ground truth.

CASE_POOL: List[Dict[str, Any]] = [
    {
        "id": "PT-001", "age": 28, "gender": "Female",
        "history": "No significant history",
        "chief_complaint": "Mild headache and runny nose for 2 days",
        "symptom_layers": [
            "Nasal congestion, mild sore throat, occasional sneezing.",
            "No fever, no body aches, appetite normal."
        ],
        "vital_layers": [
            "Temp: 37.0°C, HR: 72 bpm, BP: 110/70 mmHg, SpO2: 99%",
            "All vitals stable and within normal range."
        ],
        "true_urgency": "low",
        "true_action": "self_care",
        "volatility": 0,  # no deterioration
    },
    {
        "id": "PT-002", "age": 45, "gender": "Male",
        "history": "Hypertension, smoker",
        "chief_complaint": "Persistent cough with blood-tinged sputum",
        "symptom_layers": [
            "Productive cough for 3 weeks, weight loss of 4 kg.",
            "Night sweats, fatigue, hemoptysis worsening."
        ],
        "vital_layers": [
            "Temp: 37.8°C, HR: 92 bpm, BP: 145/90 mmHg, SpO2: 94%",
            "Mild tachycardia, slightly elevated temperature, SpO2 dropping."
        ],
        "true_urgency": "medium",
        "true_action": "clinic_visit",
        "volatility": 1,
    },
    {
        "id": "PT-003", "age": 55, "gender": "Male",
        "history": "Type 2 Diabetes, previous MI",
        "chief_complaint": "Severe crushing chest pain radiating to left arm",
        "symptom_layers": [
            "Diaphoresis, nausea, feeling of impending doom.",
            "Pain worsening, jaw numbness, shortness of breath."
        ],
        "vital_layers": [
            "Temp: 36.5°C, HR: 110 bpm, BP: 160/95 mmHg, SpO2: 92%",
            "HR rising to 130, BP dropping to 90/60 — CRITICAL."
        ],
        "true_urgency": "high",
        "true_action": "emergency_room",
        "volatility": 2,
    },
    {
        "id": "PT-004", "age": 6, "gender": "Male",
        "history": "No significant history",
        "chief_complaint": "High fever and ear pain since yesterday",
        "symptom_layers": [
            "Pulling at right ear, irritable, decreased appetite.",
            "Mild neck stiffness, crying when lying flat."
        ],
        "vital_layers": [
            "Temp: 39.5°C, HR: 115 bpm, BP: 95/60 mmHg, SpO2: 97%",
            "Temp rising to 39.8°C, slight dehydration signs."
        ],
        "true_urgency": "medium",
        "true_action": "clinic_visit",
        "volatility": 1,
    },
    {
        "id": "PT-005", "age": 70, "gender": "Female",
        "history": "Atrial fibrillation, on blood thinners",
        "chief_complaint": "Sudden weakness on left side of body",
        "symptom_layers": [
            "Slurred speech, facial droop on left, confusion.",
            "Unable to lift left arm, visual disturbance."
        ],
        "vital_layers": [
            "Temp: 36.8°C, HR: 88 irregular, BP: 185/100 mmHg, SpO2: 96%",
            "BP spiking to 200/110, consciousness deteriorating."
        ],
        "true_urgency": "high",
        "true_action": "emergency_room",
        "volatility": 2,
    },
    {
        "id": "PT-006", "age": 35, "gender": "Female",
        "history": "Seasonal allergies",
        "chief_complaint": "Itchy eyes and sneezing for a week",
        "symptom_layers": [
            "Watery eyes, nasal congestion, post-nasal drip.",
            "No fever, no shortness of breath, symptoms worse outdoors."
        ],
        "vital_layers": [
            "Temp: 36.9°C, HR: 68 bpm, BP: 118/75 mmHg, SpO2: 99%",
            "All vitals perfectly normal."
        ],
        "true_urgency": "low",
        "true_action": "self_care",
        "volatility": 0,
    },
]

# Per-task deterministic case assignment for reproducibility
TASK_CASES = {
    "direct_triage": [CASE_POOL[0], CASE_POOL[2], CASE_POOL[5]],       # easy: clear-cut
    "investigative_triage": [CASE_POOL[1], CASE_POOL[3], CASE_POOL[4]], # medium: need investigation
    "time_critical_triage": [CASE_POOL[2], CASE_POOL[4], CASE_POOL[1]], # hard: volatile patients
}

TASK_INSTRUCTIONS = {
    "direct_triage": (
        "You have full patient information. Classify the urgency level "
        "(low/medium/high) and recommend an action (self_care/clinic_visit/emergency_room). "
        "Use action_type='triage' to submit your decision."
    ),
    "investigative_triage": (
        "You only see the chief complaint. Use action_type='investigate' with "
        "investigation_target='symptoms' or 'vitals' to gather information. "
        "Each investigation costs time (penalty). Once ready, use action_type='triage' "
        "to submit your decision."
    ),
    "time_critical_triage": (
        "URGENT: The patient may deteriorate rapidly. You must balance investigation "
        "against the risk of delay. Each step may worsen high-volatility patients. "
        "Investigate only if critical, then triage quickly. Delayed triage on unstable "
        "patients will be severely penalized."
    ),
}


# ─────────────────────── Environment ───────────────────────

class MedicalTriageEnv:
    """
    OpenEnv-compliant Medical Triage RL Environment.

    API:
      reset()  → TriageObservation
      step(action: TriageAction) → (TriageObservation, TriageReward, bool, dict)
      state()  → dict
      close()  → None
    """

    def __init__(self, task_name: str = "direct_triage", case_index: int = 0):
        assert task_name in TASK_INSTRUCTIONS, (
            f"Unknown task '{task_name}'. Choose from: {list(TASK_INSTRUCTIONS.keys())}"
        )
        self.task_name: str = task_name
        self.case_index: int = case_index % len(TASK_CASES[task_name])
        self.max_steps: int = 5
        self.task_instructions: str = TASK_INSTRUCTIONS[task_name]

        # Episode state (set on reset)
        self.current_case: Optional[Dict[str, Any]] = None
        self.step_number: int = 0
        self.done: bool = False
        self.investigation_penalty: float = 0.0
        self.symptoms_discovered: int = 0
        self.vitals_discovered: int = 0
        self.discovered_symptoms: str = "Not yet investigated"
        self.discovered_vitals: str = "Not yet investigated"
        self.chief_complaint: str = ""
        self.rewards_history: List[float] = []
        self.deterioration_applied: bool = False

    # ──────── SCORE CLAMPING ────────
    # ALL scores are clamped to (0.01, 0.99) so they never hit 0.0 or 1.0.

    @staticmethod
    def _safe_score(raw: float) -> float:
        """Clamp to the strict open interval (0.01, 0.99)."""
        return round(max(0.01, min(0.99, raw)), 4)

    # ──────── CORE API ────────

    def reset(self) -> TriageObservation:
        """Reset the environment to a fresh episode."""
        cases = TASK_CASES[self.task_name]
        self.current_case = copy.deepcopy(cases[self.case_index % len(cases)])
        self.step_number = 0
        self.done = False
        self.investigation_penalty = 0.0
        self.symptoms_discovered = 0
        self.vitals_discovered = 0
        self.rewards_history = []
        self.deterioration_applied = False
        self.chief_complaint = self.current_case["chief_complaint"]

        if self.task_name == "direct_triage":
            # Easy: all information is visible immediately
            self.discovered_symptoms = " | ".join(self.current_case["symptom_layers"])
            self.discovered_vitals = " | ".join(self.current_case["vital_layers"])
            self.symptoms_discovered = len(self.current_case["symptom_layers"])
            self.vitals_discovered = len(self.current_case["vital_layers"])
        else:
            self.discovered_symptoms = "Not yet investigated"
            self.discovered_vitals = "Not yet investigated"

        return self._get_observation()

    def step(self, action: TriageAction) -> Tuple[TriageObservation, TriageReward, bool, dict]:
        """Execute one step."""
        if self.done:
            obs = self._get_observation()
            r = self._safe_score(0.01)
            return obs, TriageReward(score=r, reward=r, feedback="Episode already ended."), True, {"score": r}

        if self.current_case is None:
            obs_reset = self.reset()
            # Recursively call step after reset
            return self.step(action)

        self.step_number += 1

        # Apply deterioration for time_critical_triage
        if self.task_name == "time_critical_triage" and self.step_number > 1:
            self._apply_deterioration()

        # Timeout guard
        if self.step_number > self.max_steps:
            self.done = True
            r = self._safe_score(0.15 - self.investigation_penalty)
            self.rewards_history.append(r)
            return (
                self._get_observation(),
                TriageReward(score=r, reward=r, feedback="Ran out of steps. Episode terminated."),
                True,
                {"score": r, "reason": "timeout"},
            )

        # ─── Handle INVESTIGATE ───
        if action.action_type == "investigate":
            return self._handle_investigate(action)

        # ─── Handle TRIAGE ───
        elif action.action_type == "triage":
            return self._handle_triage(action)

        # Unknown action type (shouldn't happen with Pydantic validation)
        self.done = True
        r = self._safe_score(0.05)
        self.rewards_history.append(r)
        return (
            self._get_observation(),
            TriageReward(score=r, reward=r, feedback="Invalid action type."),
            True,
            {"score": r},
        )

    def state(self) -> dict:
        """Return current environment state (for debugging / OpenEnv compliance)."""
        return {
            "task_name": self.task_name,
            "step_number": self.step_number,
            "done": self.done,
            "max_steps": self.max_steps,
            "investigation_penalty": round(self.investigation_penalty, 4),
            "symptoms_discovered": self.symptoms_discovered,
            "vitals_discovered": self.vitals_discovered,
            "rewards_history": self.rewards_history,
            "current_patient": self.current_case["id"] if self.current_case else None,
            "deterioration_applied": self.deterioration_applied,
        }

    def close(self):
        """Clean up resources."""
        self.current_case = None
        self.done = True

    # ──────── INTERNAL HELPERS ────────

    def _get_observation(self) -> TriageObservation:
        if self.current_case is None:
            return TriageObservation(
                patient_id="NONE", age=0, gender="Unknown",
                medical_history="N/A", chief_complaint="N/A",
                discovered_symptoms="N/A", discovered_vitals="N/A",
                step_number=0, max_steps=self.max_steps,
                task_instructions=self.task_instructions,
            )
        return TriageObservation(
            patient_id=self.current_case["id"],
            age=self.current_case["age"],
            gender=self.current_case["gender"],
            medical_history=self.current_case["history"],
            chief_complaint=self.chief_complaint,
            discovered_symptoms=self.discovered_symptoms,
            discovered_vitals=self.discovered_vitals,
            step_number=self.step_number,
            max_steps=self.max_steps,
            task_instructions=self.task_instructions,
        )

    def _handle_investigate(self, action: TriageAction):
        """Process an investigation action."""
        target = action.investigation_target
        feedback_parts = []

        if target == "symptoms":
            if self.symptoms_discovered < len(self.current_case["symptom_layers"]):
                layer = self.current_case["symptom_layers"][self.symptoms_discovered]
                self.symptoms_discovered += 1
                # Build cumulative display
                visible = self.current_case["symptom_layers"][:self.symptoms_discovered]
                self.discovered_symptoms = " | ".join(visible)
                feedback_parts.append(f"Symptoms investigated: {layer}")
            else:
                feedback_parts.append("No new symptom information available.")

        elif target == "vitals":
            if self.vitals_discovered < len(self.current_case["vital_layers"]):
                layer = self.current_case["vital_layers"][self.vitals_discovered]
                self.vitals_discovered += 1
                visible = self.current_case["vital_layers"][:self.vitals_discovered]
                self.discovered_vitals = " | ".join(visible)
                feedback_parts.append(f"Vitals checked: {layer}")
            else:
                feedback_parts.append("No new vital sign data available.")
        else:
            feedback_parts.append("Invalid investigation target — specify 'symptoms' or 'vitals'.")

        # Investigation penalty (time cost)
        step_penalty = 0.06
        if self.task_name == "time_critical_triage":
            step_penalty = 0.10  # higher cost for time-critical cases
        self.investigation_penalty += step_penalty

        # Small positive reward for gathering information
        info_bonus = 0.05 if (self.symptoms_discovered + self.vitals_discovered) > 0 else 0.02
        r = self._safe_score(info_bonus)
        self.rewards_history.append(r)

        feedback = " ".join(feedback_parts) + f" (time penalty: -{step_penalty:.2f})"
        return (
            self._get_observation(),
            TriageReward(score=r, reward=r, feedback=feedback),
            False,
            {"score": r, "penalty_so_far": self.investigation_penalty},
        )

    def _handle_triage(self, action: TriageAction):
        """Process a triage (final decision) action."""
        self.done = True
        true_u = self.current_case["true_urgency"]
        true_a = self.current_case["true_action"]

        # Urgency match: 0.45 points
        urgency_score = 0.0
        if action.urgency_level == true_u:
            urgency_score = 0.45
        elif self._urgency_distance(action.urgency_level, true_u) == 1:
            urgency_score = 0.15  # partial credit for being 1 level off

        # Action match: 0.45 points
        action_score = 0.0
        if action.recommended_action == true_a:
            action_score = 0.45
        elif self._action_distance(action.recommended_action, true_a) == 1:
            action_score = 0.15  # partial credit

        # Investigation bonus: reward agents who investigated before triaging
        investigation_bonus = 0.0
        if self.task_name != "direct_triage":
            total_info = self.symptoms_discovered + self.vitals_discovered
            if total_info >= 2:
                investigation_bonus = 0.05  # gathered good info
            elif total_info == 1:
                investigation_bonus = 0.02

        # Deterioration penalty (for time_critical)
        deterioration_penalty = 0.0
        if self.deterioration_applied:
            deterioration_penalty = 0.10

        raw_score = (
            urgency_score
            + action_score
            + investigation_bonus
            - self.investigation_penalty
            - deterioration_penalty
        )

        final_score = self._safe_score(raw_score)
        self.rewards_history.append(final_score)

        # Build feedback
        feedback_parts = []
        if urgency_score == 0.45:
            feedback_parts.append("✓ Urgency correct")
        elif urgency_score > 0:
            feedback_parts.append(f"~ Urgency partially correct (expected {true_u})")
        else:
            feedback_parts.append(f"✗ Urgency wrong (expected {true_u})")

        if action_score == 0.45:
            feedback_parts.append("✓ Action correct")
        elif action_score > 0:
            feedback_parts.append(f"~ Action partially correct (expected {true_a})")
        else:
            feedback_parts.append(f"✗ Action wrong (expected {true_a})")

        if self.investigation_penalty > 0:
            feedback_parts.append(f"Investigation penalty: -{self.investigation_penalty:.2f}")
        if deterioration_penalty > 0:
            feedback_parts.append(f"Deterioration penalty: -{deterioration_penalty:.2f}")

        feedback = " | ".join(feedback_parts)

        info = {
            "score": final_score,
            "true_urgency": true_u,
            "true_action": true_a,
            "urgency_score": urgency_score,
            "action_score": action_score,
            "investigation_bonus": investigation_bonus,
            "investigation_penalty": self.investigation_penalty,
            "deterioration_penalty": deterioration_penalty,
        }

        return (
            self._get_observation(),
            TriageReward(score=final_score, reward=final_score, feedback=feedback),
            True,
            info,
        )

    def _apply_deterioration(self):
        """For time_critical: high volatility patients get worse over time."""
        if self.current_case is None:
            return
        volatility = self.current_case.get("volatility", 0)
        if volatility >= 2 and self.step_number >= 3 and not self.deterioration_applied:
            # Escalate the ground truth for the worst cases
            self.deterioration_applied = True
            if self.current_case["true_urgency"] == "medium":
                self.current_case["true_urgency"] = "high"
                self.current_case["true_action"] = "emergency_room"

    @staticmethod
    def _urgency_distance(a: Optional[str], b: str) -> int:
        levels = ["low", "medium", "high"]
        if a is None or a not in levels:
            return 3
        return abs(levels.index(a) - levels.index(b))

    @staticmethod
    def _action_distance(a: Optional[str], b: str) -> int:
        actions = ["self_care", "clinic_visit", "emergency_room"]
        if a is None or a not in actions:
            return 3
        return abs(actions.index(a) - actions.index(b))
