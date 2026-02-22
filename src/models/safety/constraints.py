"""
Safety Constraints
Implements all constraint types for healthcare RL safety
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, List
import numpy as np


class Constraint(ABC):
    """Base class for all constraints"""
    
    @abstractmethod
    def check(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if state-action pair satisfies constraint
        
        Args:
            state: Patient state dictionary
            action: Proposed action dictionary
            
        Returns:
            (is_satisfied, violation_message)
        """
        pass
    
    @abstractmethod
    def get_constraint_name(self) -> str:
        """Return name of constraint for logging"""
        pass


class DosageConstraint(Constraint):
    """Check medication dosage within safe range"""
    
    def __init__(self, drug_limits: Dict[str, Tuple[float, float]]):
        """
        Args:
            drug_limits: Dictionary mapping drug names to (min_dose, max_dose)
                        e.g., {'insulin': (0.0, 100.0)}
        """
        self.limits = drug_limits
    
    def check(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if dosage is within limits
        
        Args:
            state: Patient state (may contain age for age-adjusted limits)
            action: Action containing 'medication_type' and 'dosage'
            
        Returns:
            (is_safe, violation_message)
        """
        # If no medication action, pass
        if 'medication_type' not in action or 'dosage' not in action:
            return True, None
        
        drug = action['medication_type']
        dose = action['dosage']
        
        # Check if drug is known
        if drug not in self.limits:
            return False, f"Unknown drug: {drug}"
        
        min_dose, max_dose = self.limits[drug]
        
        # Age adjustment (optional)
        if 'age' in state:
            age = state['age']
            if age < 18:  # Pediatric adjustment
                max_dose = max_dose * 0.5
            elif age > 65:  # Geriatric adjustment
                max_dose = max_dose * 0.75
        
        # Check dosage bounds
        if not (min_dose <= dose <= max_dose):
            return False, f"Dosage {dose:.2f} for {drug} outside safe range [{min_dose:.2f}, {max_dose:.2f}]"
        
        return True, None
    
    def get_constraint_name(self) -> str:
        return "DosageConstraint"


class PhysiologicalConstraint(Constraint):
    """Check if predicted next state is within safe physiological ranges"""
    
    def __init__(self, safe_ranges: Dict[str, Tuple[float, float]], 
                 dynamics_model: Optional[Any] = None):
        """
        Args:
            safe_ranges: Dictionary mapping variables to (min_val, max_val)
                        e.g., {'glucose': (70.0, 200.0)}
            dynamics_model: Optional model to predict next state
        """
        self.safe_ranges = safe_ranges
        self.dynamics_model = dynamics_model
    
    def _predict_next_state(self, state: Dict[str, Any], 
                           action: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict next state given current state and action
        
        If dynamics_model is available, use it. Otherwise, use heuristics.
        
        Args:
            state: Current patient state
            action: Proposed action
            
        Returns:
            predicted_next_state: Predicted state after action
        """
        if self.dynamics_model is not None:
            # Use learned dynamics model
            return self.dynamics_model.predict(state, action)
        
        # Simple heuristic prediction
        next_state = {}
        
        # Glucose response to insulin
        if 'glucose' in state and 'medication_type' in action:
            current_glucose = state['glucose']
            
            if action.get('medication_type') == 'insulin':
                # Insulin decreases glucose
                dose = action.get('dosage', 0)
                # Simplified linear model: 1 unit insulin decreases glucose by ~30 mg/dL
                glucose_drop = dose * 30.0
                predicted_glucose = current_glucose - glucose_drop
                next_state['glucose'] = max(0, predicted_glucose)
            elif action.get('medication_type') == 'metformin':
                # Metformin has slower effect
                dose = action.get('dosage', 0)
                glucose_drop = dose * 0.02  # Much slower effect
                predicted_glucose = current_glucose - glucose_drop
                next_state['glucose'] = max(0, predicted_glucose)
            else:
                next_state['glucose'] = current_glucose
        
        # Blood pressure response
        if 'blood_pressure_systolic' in state and 'medication_type' in action:
            current_bp = state['blood_pressure_systolic']
            
            if action.get('medication_type') == 'lisinopril':
                # ACE inhibitor decreases BP
                dose = action.get('dosage', 0)
                bp_drop = dose * 0.5
                next_state['blood_pressure_systolic'] = current_bp - bp_drop
            else:
                next_state['blood_pressure_systolic'] = current_bp
        
        # Copy other variables (assuming they don't change immediately)
        for key in self.safe_ranges.keys():
            if key not in next_state:
                next_state[key] = state.get(key, 0)
        
        return next_state
    
    def check(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if predicted next state is within safe ranges
        
        Args:
            state: Current patient state
            action: Proposed action
            
        Returns:
            (is_safe, violation_message)
        """
        predicted_next_state = self._predict_next_state(state, action)
        violations = []
        
        for var, (min_val, max_val) in self.safe_ranges.items():
            if var in predicted_next_state:
                value = predicted_next_state[var]
                if not (min_val <= value <= max_val):
                    violations.append(
                        f"{var}={value:.2f} outside safe range [{min_val}, {max_val}]"
                    )
        
        if violations:
            return False, "; ".join(violations)
        
        return True, None
    
    def get_constraint_name(self) -> str:
        return "PhysiologicalConstraint"


class ContraindicationConstraint(Constraint):
    """Check for drug allergies and interactions"""
    
    def __init__(self, contraindication_db: Optional[Dict] = None):
        """
        Args:
            contraindication_db: Database of drug interactions and allergies
        """
        self.db = contraindication_db or self._default_contraindications()
    
    def _default_contraindications(self) -> Dict:
        """Default contraindication database"""
        return {
            'interactions': {
                'insulin': ['sulfonamides'],
                'metformin': ['cimetidine'],
                'warfarin': ['aspirin', 'nsaids'],
                'lisinopril': ['potassium_supplements'],
                'atorvastatin': ['gemfibrozil'],
            },
            'age_restrictions': {
                'aspirin': {'min_age': 18},  # Reye's syndrome risk
                'metformin': {'min_age': 10},
            },
            'condition_contraindications': {
                'metformin': ['kidney_disease'],
                'nsaids': ['kidney_disease', 'heart_failure'],
            }
        }
    
    def check(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check for contraindications
        
        Args:
            state: Patient state (must contain 'allergies', 'current_medications', 'age', 'conditions')
            action: Action containing 'medication_type'
            
        Returns:
            (is_safe, violation_message)
        """
        if 'medication_type' not in action:
            return True, None
        
        prescribed_drug = action['medication_type']
        
        # Check allergies
        patient_allergies = state.get('allergies', [])
        if prescribed_drug in patient_allergies:
            return False, f"Patient allergic to {prescribed_drug}"
        
        # Check drug interactions
        current_meds = state.get('current_medications', [])
        if prescribed_drug in self.db['interactions']:
            contraindicated_drugs = self.db['interactions'][prescribed_drug]
            interactions = [drug for drug in contraindicated_drugs if drug in current_meds]
            if interactions:
                return False, f"Drug interaction: {prescribed_drug} with {interactions}"
        
        # Check age restrictions
        if prescribed_drug in self.db['age_restrictions']:
            restrictions = self.db['age_restrictions'][prescribed_drug]
            patient_age = state.get('age', 0)
            
            if 'min_age' in restrictions and patient_age < restrictions['min_age']:
                return False, f"{prescribed_drug} contraindicated for age {patient_age} (min: {restrictions['min_age']})"
            
            if 'max_age' in restrictions and patient_age > restrictions['max_age']:
                return False, f"{prescribed_drug} contraindicated for age {patient_age} (max: {restrictions['max_age']})"
        
        # Check condition contraindications
        if prescribed_drug in self.db['condition_contraindications']:
            contraindicated_conditions = self.db['condition_contraindications'][prescribed_drug]
            patient_conditions = state.get('conditions', [])
            
            conditions_present = [cond for cond in contraindicated_conditions if cond in patient_conditions]
            if conditions_present:
                return False, f"{prescribed_drug} contraindicated with conditions: {conditions_present}"
        
        return True, None
    
    def get_constraint_name(self) -> str:
        return "ContraindicationConstraint"


class FrequencyConstraint(Constraint):
    """Check frequency limits for appointments and reminders"""
    
    def __init__(self, max_reminders_per_week: int = 7,
                 min_appointment_interval_days: int = 3):
        """
        Args:
            max_reminders_per_week: Maximum number of reminders per week
            min_appointment_interval_days: Minimum days between appointments
        """
        self.max_reminders = max_reminders_per_week
        self.min_interval = min_appointment_interval_days
    
    def check(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check frequency constraints
        
        Args:
            state: Patient state (may contain last appointment date)
            action: Action containing 'reminder_frequency' and/or 'next_appointment_days'
            
        Returns:
            (is_safe, violation_message)
        """
        # Check reminder frequency
        if 'reminder_frequency' in action:
            freq = action['reminder_frequency']
            if freq > self.max_reminders:
                return False, f"Reminder frequency {freq} exceeds maximum {self.max_reminders}/week"
        
        # Check appointment scheduling
        if 'next_appointment_days' in action:
            days = action['next_appointment_days']
            if days < self.min_interval:
                return False, f"Appointment interval {days} days less than minimum {self.min_interval}"
        
        return True, None
    
    def get_constraint_name(self) -> str:
        return "FrequencyConstraint"
