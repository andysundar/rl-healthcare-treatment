"""
Cost-Effectiveness Reward Function
Penalizes high-cost interventions to encourage cost-effective care.
"""

from typing import Dict, Any
import numpy as np
from .base_reward import BaseRewardFunction


class CostEffectivenessReward(BaseRewardFunction):
    """
    Reward function for cost-effectiveness.
    
    Penalizes:
    1. Expensive medications
    2. Frequent appointments
    3. Emergency visits
    4. Hospitalizations
    
    Clinical Rationale:
    - Healthcare costs are a major concern
    - Cost-effective care improves access
    - Should balance cost with outcomes (not minimize cost alone)
    
    Note: Returns negative values (cost as penalty).
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize cost-effectiveness reward.
        
        Args:
            config: Configuration with cost parameters
        """
        super().__init__(config)
        
        # Medication costs (per unit/dose)
        self.medication_costs = getattr(config, 'medication_costs', {
            'insulin': 0.50,  # $ per unit
            'metformin': 0.10,  # $ per dose
            'glipizide': 0.15,  # $ per dose
            'statin': 0.20,  # $ per dose
            'ace_inhibitor': 0.25  # $ per dose
        })
        
        # Service costs
        self.appointment_cost = getattr(config, 'appointment_cost', 100.0)
        self.lab_test_cost = getattr(config, 'lab_test_cost', 50.0)
        self.emergency_visit_cost = getattr(config, 'emergency_visit_cost', 5000.0)
        self.hospitalization_cost_per_day = getattr(config, 'hospitalization_cost_per_day', 2000.0)
        self.icu_cost_per_day = getattr(config, 'icu_cost_per_day', 10000.0)
        
        # Reminder/communication costs (minimal but non-zero)
        self.reminder_cost = getattr(config, 'reminder_cost', 0.50)
        self.phone_call_cost = getattr(config, 'phone_call_cost', 5.0)
        
        # Cost normalization factor (to scale costs to similar range as other rewards)
        self.cost_normalization = getattr(config, 'cost_normalization_factor', 1000.0)
        
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute cost-effectiveness reward (negative cost).
        
        Args:
            state: Previous patient state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Negative cost (lower cost = less negative = better)
        """
        total_cost = 0.0
        
        # Medication costs
        total_cost += self._compute_medication_costs(action)
        
        # Service costs
        total_cost += self._compute_service_costs(action)
        
        # Adverse event costs
        total_cost += self._compute_adverse_event_costs(next_state)
        
        # Communication costs
        total_cost += self._compute_communication_costs(action)
        
        # Normalize and return as negative reward
        normalized_cost = total_cost / self.cost_normalization
        return -normalized_cost
    
    def _compute_medication_costs(self, action: Dict[str, Any]) -> float:
        """
        Compute medication costs.
        
        Args:
            action: Action with medication dosages
            
        Returns:
            Total medication cost
        """
        cost = 0.0
        
        for med_name, unit_cost in self.medication_costs.items():
            dose_key = f'{med_name}_dose'
            if dose_key in action:
                dose = action[dose_key]
                cost += dose * unit_cost
        
        # Alternative: prescribed medications list
        if 'prescribed_medications' in action:
            for med in action['prescribed_medications']:
                if med in self.medication_costs:
                    # Assume standard dose if not specified
                    cost += self.medication_costs[med]
        
        return cost
    
    def _compute_service_costs(self, action: Dict[str, Any]) -> float:
        """
        Compute healthcare service costs.
        
        Args:
            action: Action with service requests
            
        Returns:
            Total service cost
        """
        cost = 0.0
        
        # Appointment scheduling
        if action.get('schedule_appointment', False):
            cost += self.appointment_cost
        
        # Lab tests ordered
        num_lab_tests = action.get('num_lab_tests', 0)
        cost += num_lab_tests * self.lab_test_cost
        
        # Specialist referral (treated as additional appointment)
        if action.get('specialist_referral', False):
            cost += self.appointment_cost * 2  # Specialist costs more
        
        return cost
    
    def _compute_adverse_event_costs(self, state: Dict[str, Any]) -> float:
        """
        Compute costs from adverse events.
        
        Args:
            state: Patient state with adverse event indicators
            
        Returns:
            Adverse event costs
        """
        cost = 0.0
        
        # Emergency visit
        if state.get('emergency_visit', False):
            cost += self.emergency_visit_cost
        
        # Hospitalization
        if state.get('hospitalized', False):
            days = state.get('hospital_days', 1)
            cost += days * self.hospitalization_cost_per_day
        
        # ICU admission
        if state.get('icu_admission', False):
            days = state.get('icu_days', 1)
            cost += days * self.icu_cost_per_day
        
        return cost
    
    def _compute_communication_costs(self, action: Dict[str, Any]) -> float:
        """
        Compute communication and reminder costs.
        
        Args:
            action: Action with communication plans
            
        Returns:
            Communication costs
        """
        cost = 0.0
        
        # Reminders
        num_reminders = action.get('num_reminders', 0)
        cost += num_reminders * self.reminder_cost
        
        # Phone calls
        if action.get('phone_call_scheduled', False):
            cost += self.phone_call_cost
        
        return cost
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of cost components.
        
        Returns:
            Dictionary with cost breakdown
        """
        components = {
            'cost_medication': -self._compute_medication_costs(action) / self.cost_normalization,
            'cost_services': -self._compute_service_costs(action) / self.cost_normalization,
            'cost_adverse_events': -self._compute_adverse_event_costs(next_state) / self.cost_normalization,
            'cost_communication': -self._compute_communication_costs(action) / self.cost_normalization,
            'cost_total': self.compute_reward(state, action, next_state)
        }
        
        return components
    
    def compute_total_episode_cost(self, trajectory: list) -> float:
        """
        Compute total cost over an entire episode.
        
        Args:
            trajectory: List of (state, action, next_state) tuples
            
        Returns:
            Total episode cost (non-negative)
        """
        total_cost = 0.0
        
        for state, action, next_state in trajectory:
            # Get negative of reward (which is negative cost)
            cost = -self.compute_reward(state, action, next_state)
            total_cost += cost * self.cost_normalization
        
        return total_cost
    
    def compute_cost_effectiveness_ratio(
        self,
        trajectory: list,
        health_improvement: float
    ) -> float:
        """
        Compute cost per unit of health improvement.
        
        Args:
            trajectory: Episode trajectory
            health_improvement: Total health improvement (e.g., HbA1c reduction)
            
        Returns:
            Cost-effectiveness ratio (cost per unit improvement)
        """
        total_cost = self.compute_total_episode_cost(trajectory)
        
        if health_improvement <= 0:
            return float('inf')  # Infinite if no improvement
        
        return total_cost / health_improvement
