"""
Safety Penalty Function
Penalizes unsafe physiological states, dangerous actions, and adverse events.
"""

from typing import Dict, Any, List, Set
import numpy as np
from .base_reward import BaseRewardFunction


class SafetyPenalty(BaseRewardFunction):
    """
    Penalty function for unsafe states and actions.
    
    Penalizes:
    1. Dangerous physiological states (hypoglycemia, hypertension, etc.)
    2. Unsafe medication dosages
    3. Drug interactions
    4. Adverse events (hospitalizations, complications)
    
    Clinical Rationale:
    - Safety is paramount in healthcare
    - Some states are immediately dangerous (e.g., severe hypoglycemia)
    - Drug interactions can be fatal
    - Emergency events should be strongly penalized
    
    Note: This function returns negative values (penalties).
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize safety penalty.
        
        Args:
            config: Configuration with safety thresholds and penalties
        """
        super().__init__(config)
        
        # Physiological danger thresholds
        self.glucose_severe_low = getattr(config, 'severe_hypoglycemia_threshold', 60.0)
        self.glucose_severe_high = getattr(config, 'severe_hyperglycemia_threshold', 300.0)
        self.glucose_moderate_low = getattr(config, 'moderate_hypoglycemia_threshold', 70.0)
        self.glucose_moderate_high = getattr(config, 'moderate_hyperglycemia_threshold', 200.0)
        
        self.systolic_severe_high = getattr(config, 'severe_hypertension_threshold', 180.0)
        self.systolic_moderate_high = getattr(config, 'moderate_hypertension_threshold', 140.0)
        self.systolic_severe_low = getattr(config, 'severe_hypotension_threshold', 70.0)
        
        # Penalty magnitudes
        self.severe_hypoglycemia_penalty = getattr(config, 'severe_hypoglycemia_penalty', -10.0)
        self.severe_hyperglycemia_penalty = getattr(config, 'severe_hyperglycemia_penalty', -5.0)
        self.moderate_hypoglycemia_penalty = getattr(config, 'moderate_hypoglycemia_penalty', -3.0)
        self.moderate_hyperglycemia_penalty = getattr(config, 'moderate_hyperglycemia_penalty', -2.0)
        
        self.severe_hypertension_penalty = getattr(config, 'severe_hypertension_penalty', -8.0)
        self.moderate_hypertension_penalty = getattr(config, 'moderate_hypertension_penalty', -3.0)
        self.severe_hypotension_penalty = getattr(config, 'severe_hypotension_penalty', -7.0)
        
        self.drug_interaction_penalty = getattr(config, 'drug_interaction_penalty', -7.0)
        self.overdose_penalty = getattr(config, 'overdose_penalty', -8.0)
        self.contraindication_penalty = getattr(config, 'contraindication_penalty', -10.0)
        
        self.emergency_visit_penalty = getattr(config, 'emergency_visit_penalty', -20.0)
        self.hospitalization_penalty = getattr(config, 'hospitalization_penalty', -15.0)
        self.icu_admission_penalty = getattr(config, 'icu_admission_penalty', -25.0)
        
        # Maximum safe dosages (example values)
        self.max_safe_dosages = getattr(config, 'max_safe_dosages', {
            'insulin': 100.0,  # units
            'metformin': 2000.0,  # mg
            'glipizide': 20.0  # mg
        })
        
        # Known dangerous drug interactions
        self.dangerous_interactions = getattr(config, 'dangerous_interactions', set())
        
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute safety penalty (always <= 0).
        
        Args:
            state: Previous patient state
            action: Action taken
            next_state: Resulting patient state
            
        Returns:
            Safety penalty (negative or zero)
        """
        total_penalty = 0.0
        
        # Physiological safety penalties
        total_penalty += self._compute_physiological_penalties(next_state)
        
        # Medication safety penalties
        total_penalty += self._compute_medication_penalties(state, action, next_state)
        
        # Adverse event penalties
        total_penalty += self._compute_adverse_event_penalties(next_state)
        
        return total_penalty
    
    def _compute_physiological_penalties(self, state: Dict[str, Any]) -> float:
        """
        Compute penalties for dangerous physiological states.
        
        Args:
            state: Patient state with vitals
            
        Returns:
            Penalty (negative or zero)
        """
        penalty = 0.0
        
        # Glucose safety
        glucose = state.get('glucose')
        if glucose is not None:
            if glucose < self.glucose_severe_low:
                penalty += self.severe_hypoglycemia_penalty
            elif glucose < self.glucose_moderate_low:
                penalty += self.moderate_hypoglycemia_penalty
            elif glucose > self.glucose_severe_high:
                penalty += self.severe_hyperglycemia_penalty
            elif glucose > self.glucose_moderate_high:
                penalty += self.moderate_hyperglycemia_penalty
        
        # Blood pressure safety
        systolic_bp = state.get('systolic_bp')
        if systolic_bp is not None:
            if systolic_bp > self.systolic_severe_high:
                penalty += self.severe_hypertension_penalty
            elif systolic_bp > self.systolic_moderate_high:
                penalty += self.moderate_hypertension_penalty
            elif systolic_bp < self.systolic_severe_low:
                penalty += self.severe_hypotension_penalty
        
        return penalty
    
    def _compute_medication_penalties(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute penalties for unsafe medication actions.
        
        Args:
            state: Previous state
            action: Medication action
            next_state: Resulting state
            
        Returns:
            Penalty (negative or zero)
        """
        penalty = 0.0
        
        # Check for overdosing
        for med_name, max_safe_dose in self.max_safe_dosages.items():
            dose_key = f'{med_name}_dose'
            if dose_key in action:
                dose = action[dose_key]
                if dose > max_safe_dose:
                    # Penalty proportional to how much over safe limit
                    excess = (dose - max_safe_dose) / max_safe_dose
                    penalty += self.overdose_penalty * (1 + excess)
        
        # Check for contraindications
        patient_allergies = state.get('allergies', [])
        patient_conditions = state.get('conditions', [])
        
        for med_name in action.get('prescribed_medications', []):
            # Allergy contraindication
            if med_name in patient_allergies:
                penalty += self.contraindication_penalty
            
            # Condition contraindication
            if self._has_contraindication(med_name, patient_conditions):
                penalty += self.contraindication_penalty
        
        # Check for drug interactions
        current_meds = state.get('current_medications', [])
        new_meds = action.get('prescribed_medications', [])
        
        for new_med in new_meds:
            for current_med in current_meds:
                interaction_key = tuple(sorted([new_med, current_med]))
                if interaction_key in self.dangerous_interactions:
                    penalty += self.drug_interaction_penalty
        
        return penalty
    
    def _compute_adverse_event_penalties(self, state: Dict[str, Any]) -> float:
        """
        Compute penalties for adverse events.
        
        Args:
            state: Patient state with adverse event indicators
            
        Returns:
            Penalty (negative or zero)
        """
        penalty = 0.0
        
        # Emergency visit
        if state.get('emergency_visit', False):
            penalty += self.emergency_visit_penalty
        
        # Hospitalization
        if state.get('hospitalized', False):
            penalty += self.hospitalization_penalty
        
        # ICU admission
        if state.get('icu_admission', False):
            penalty += self.icu_admission_penalty
        
        # Generic adverse events with severity
        if state.get('adverse_event_occurred', False):
            severity = state.get('adverse_event_severity', 1.0)
            penalty += -10.0 * severity
        
        return penalty
    
    def _has_contraindication(
        self,
        medication: str,
        conditions: List[str]
    ) -> bool:
        """
        Check if medication is contraindicated for patient conditions.
        
        Args:
            medication: Medication name
            conditions: List of patient conditions
            
        Returns:
            True if contraindicated
        """
        # Example contraindications (should be comprehensive in production)
        contraindications = {
            'metformin': ['severe_kidney_disease', 'liver_failure'],
            'insulin': ['severe_hypoglycemia_history'],
            'glipizide': ['severe_liver_disease']
        }
        
        contraindicated_conditions = contraindications.get(medication, [])
        return any(cond in conditions for cond in contraindicated_conditions)
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of safety penalties.
        
        Returns:
            Dictionary with penalty breakdown
        """
        components = {
            'safety_physiological': self._compute_physiological_penalties(next_state),
            'safety_medication': self._compute_medication_penalties(state, action, next_state),
            'safety_adverse_events': self._compute_adverse_event_penalties(next_state),
            'safety_total': self.compute_reward(state, action, next_state)
        }
        
        return components
    
    def is_safe_state(self, state: Dict[str, Any]) -> bool:
        """
        Check if a state is physiologically safe.
        
        Args:
            state: Patient state
            
        Returns:
            True if state is safe
        """
        penalty = self._compute_physiological_penalties(state)
        return penalty == 0.0
    
    def add_drug_interaction(self, drug1: str, drug2: str) -> None:
        """
        Add a dangerous drug interaction to monitor.
        
        Args:
            drug1: First medication
            drug2: Second medication
        """
        interaction_key = tuple(sorted([drug1, drug2]))
        self.dangerous_interactions.add(interaction_key)
