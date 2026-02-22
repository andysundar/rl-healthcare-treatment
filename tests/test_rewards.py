"""
Comprehensive Test Suite for Reward Functions
Tests all reward components with realistic clinical scenarios.
"""

import pytest
import numpy as np
from typing import Dict, Any

from .reward_config import RewardConfig
from .adherence_reward import AdherenceReward
from .health_reward import HealthOutcomeReward
from .safety_reward import SafetyPenalty
from .cost_reward import CostEffectivenessReward
from .composite_reward import CompositeRewardFunction
from .reward_shaping import (
    normalize_reward,
    clip_reward,
    potential_based_shaping,
    health_potential_function
)


# ========== Test Fixtures ==========

@pytest.fixture
def config():
    """Create default reward config."""
    return RewardConfig()


@pytest.fixture
def good_adherence_scenario():
    """Scenario: Patient with good medication adherence."""
    state = {
        'adherence_score': 0.85,
        'adherence_streak_days': 45,
        'glucose': 110.0,
        'systolic_bp': 115.0,
        'diastolic_bp': 75.0
    }
    
    action = {
        'insulin_dose': 20.0,
        'num_reminders': 1
    }
    
    next_state = {
        'adherence_score': 0.90,
        'adherence_streak_days': 46,
        'glucose': 105.0,
        'systolic_bp': 112.0,
        'diastolic_bp': 72.0
    }
    
    return state, action, next_state


@pytest.fixture
def poor_adherence_scenario():
    """Scenario: Patient with poor medication adherence."""
    state = {
        'adherence_score': 0.40,
        'adherence_streak_days': 0,
        'glucose': 180.0,
        'systolic_bp': 145.0
    }
    
    action = {
        'insulin_dose': 30.0,
        'num_reminders': 3
    }
    
    next_state = {
        'adherence_score': 0.35,
        'adherence_streak_days': 0,
        'glucose': 190.0,
        'systolic_bp': 148.0
    }
    
    return state, action, next_state


@pytest.fixture
def hypoglycemia_scenario():
    """Scenario: Patient experiencing hypoglycemia (dangerous)."""
    state = {
        'glucose': 75.0,
        'adherence_score': 0.8
    }
    
    action = {
        'insulin_dose': 25.0
    }
    
    next_state = {
        'glucose': 55.0,  # Severe hypoglycemia
        'adherence_score': 0.8
    }
    
    return state, action, next_state


@pytest.fixture
def emergency_scenario():
    """Scenario: Patient requires emergency visit."""
    state = {
        'glucose': 280.0,
        'systolic_bp': 175.0
    }
    
    action = {
        'insulin_dose': 40.0
    }
    
    next_state = {
        'glucose': 320.0,
        'systolic_bp': 185.0,
        'emergency_visit': True,
        'adverse_event_occurred': True,
        'adverse_event_severity': 2.0
    }
    
    return state, action, next_state


@pytest.fixture
def drug_interaction_scenario():
    """Scenario: Dangerous drug interaction."""
    state = {
        'current_medications': ['metformin', 'statin'],
        'glucose': 140.0
    }
    
    action = {
        'prescribed_medications': ['glipizide'],  # Hypothetical interaction
        'insulin_dose': 15.0
    }
    
    next_state = {
        'current_medications': ['metformin', 'statin', 'glipizide'],
        'glucose': 130.0
    }
    
    return state, action, next_state


# ========== Adherence Reward Tests ==========

def test_adherence_reward_improvement(config, good_adherence_scenario):
    """Test adherence reward for improving adherence."""
    state, action, next_state = good_adherence_scenario
    
    adherence_reward = AdherenceReward(config)
    reward = adherence_reward.compute_reward(state, action, next_state)
    
    # Should be positive (good adherence + improvement)
    assert reward > 0, f"Expected positive reward for good adherence, got {reward}"
    
    # Check components
    components = adherence_reward.get_reward_components(state, action, next_state)
    assert components['adherence_improvement'] > 0, "Should reward improvement"
    assert components['adherence_sustained_bonus'] > 0, "Should give sustained bonus"


def test_adherence_reward_decline(config, poor_adherence_scenario):
    """Test adherence reward for declining adherence."""
    state, action, next_state = poor_adherence_scenario
    
    adherence_reward = AdherenceReward(config)
    reward = adherence_reward.compute_reward(state, action, next_state)
    
    # Should still be positive but small (low adherence)
    assert reward >= 0, f"Reward should be non-negative, got {reward}"
    assert reward < 1.0, "Reward should be low for poor adherence"
    
    # No improvement bonus
    components = adherence_reward.get_reward_components(state, action, next_state)
    assert components['adherence_improvement'] == 0, "No improvement, no bonus"


# ========== Health Outcome Reward Tests ==========

def test_health_reward_good_metrics(config, good_adherence_scenario):
    """Test health reward for good health metrics."""
    state, action, next_state = good_adherence_scenario
    
    health_reward = HealthOutcomeReward(config)
    reward = health_reward.compute_reward(state, action, next_state)
    
    # Should be positive (metrics in target range)
    assert reward > 0, f"Expected positive reward for good health, got {reward}"


def test_health_reward_improvement(config):
    """Test health reward for improving metrics."""
    state = {
        'glucose': 150.0,
        'systolic_bp': 135.0
    }
    
    action = {
        'insulin_dose': 20.0
    }
    
    next_state = {
        'glucose': 120.0,  # Improved
        'systolic_bp': 118.0  # Improved
    }
    
    health_reward = HealthOutcomeReward(config)
    reward = health_reward.compute_reward(state, action, next_state)
    
    # Should be positive (improvement toward target)
    assert reward > 0, "Should reward improvement"
    
    # Check improvement components
    components = health_reward.get_reward_components(state, action, next_state)
    assert 'health_glucose_improvement' in components
    assert components['health_glucose_improvement'] > 0


def test_health_reward_deterioration(config, poor_adherence_scenario):
    """Test health reward for deteriorating metrics."""
    state, action, next_state = poor_adherence_scenario
    
    health_reward = HealthOutcomeReward(config)
    reward = health_reward.compute_reward(state, action, next_state)
    
    # Should be negative (metrics getting worse)
    assert reward < 0, "Should penalize deteriorating health"


# ========== Safety Penalty Tests ==========

def test_safety_penalty_safe_state(config, good_adherence_scenario):
    """Test that safe states incur no penalty."""
    state, action, next_state = good_adherence_scenario
    
    safety_penalty = SafetyPenalty(config)
    penalty = safety_penalty.compute_reward(state, action, next_state)
    
    # Should be zero (no safety violations)
    assert penalty == 0.0, f"Expected zero penalty for safe state, got {penalty}"
    
    # Verify state is considered safe
    assert safety_penalty.is_safe_state(next_state)


def test_safety_penalty_hypoglycemia(config, hypoglycemia_scenario):
    """Test severe penalty for hypoglycemia."""
    state, action, next_state = hypoglycemia_scenario
    
    safety_penalty = SafetyPenalty(config)
    penalty = safety_penalty.compute_reward(state, action, next_state)
    
    # Should be large negative penalty
    assert penalty < -5.0, f"Expected severe penalty for hypoglycemia, got {penalty}"
    
    # Verify state is considered unsafe
    assert not safety_penalty.is_safe_state(next_state)


def test_safety_penalty_emergency_visit(config, emergency_scenario):
    """Test severe penalty for emergency visit."""
    state, action, next_state = emergency_scenario
    
    safety_penalty = SafetyPenalty(config)
    penalty = safety_penalty.compute_reward(state, action, next_state)
    
    # Should be very large negative penalty
    assert penalty < -15.0, "Expected severe penalty for emergency visit"
    
    # Check components
    components = safety_penalty.get_reward_components(state, action, next_state)
    assert components['safety_adverse_events'] < 0


def test_safety_penalty_overdose(config):
    """Test penalty for medication overdose."""
    state = {
        'glucose': 120.0
    }
    
    action = {
        'insulin_dose': 150.0  # Over max safe dose of 100
    }
    
    next_state = {
        'glucose': 110.0
    }
    
    safety_penalty = SafetyPenalty(config)
    penalty = safety_penalty.compute_reward(state, action, next_state)
    
    # Should penalize overdose
    assert penalty < 0, "Should penalize overdose"


# ========== Cost Reward Tests ==========

def test_cost_reward_basic_treatment(config):
    """Test cost calculation for basic treatment."""
    state = {
        'glucose': 140.0
    }
    
    action = {
        'insulin_dose': 20.0,
        'schedule_appointment': False,
        'num_reminders': 1
    }
    
    next_state = {
        'glucose': 120.0
    }
    
    cost_reward = CostEffectivenessReward(config)
    reward = cost_reward.compute_reward(state, action, next_state)
    
    # Should be small negative (only insulin + reminder cost)
    assert reward < 0, "Cost should be negative"
    assert reward > -1.0, "Basic treatment should be relatively cheap"


def test_cost_reward_expensive_scenario(config, emergency_scenario):
    """Test cost for expensive emergency scenario."""
    state, action, next_state = emergency_scenario
    
    cost_reward = CostEffectivenessReward(config)
    reward = cost_reward.compute_reward(state, action, next_state)
    
    # Should be large negative (emergency visit is expensive)
    assert reward < -2.0, "Emergency visit should be very expensive"


# ========== Composite Reward Tests ==========

def test_composite_reward_integration(config, good_adherence_scenario):
    """Test composite reward combining all components."""
    state, action, next_state = good_adherence_scenario
    
    # Create composite reward
    composite = CompositeRewardFunction(config)
    
    # Add components
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    reward = composite.compute_reward(state, action, next_state)
    
    # Good scenario should have positive total reward
    assert reward > 0, f"Expected positive total reward for good scenario, got {reward}"
    
    # Get breakdown
    breakdown = composite.get_reward_components(state, action, next_state)
    assert 'adherence_weighted' in breakdown
    assert 'health_weighted' in breakdown
    assert 'safety_weighted' in breakdown
    assert 'cost_weighted' in breakdown


def test_composite_reward_bad_scenario(config, emergency_scenario):
    """Test composite reward for bad scenario."""
    state, action, next_state = emergency_scenario
    
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    reward = composite.compute_reward(state, action, next_state)
    
    # Bad scenario should have negative total reward (safety dominates)
    assert reward < 0, "Emergency scenario should have negative total reward"


def test_composite_reward_weight_adjustment(config, good_adherence_scenario):
    """Test that changing weights affects total reward."""
    state, action, next_state = good_adherence_scenario
    
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), 1.0)
    composite.add_component('health', HealthOutcomeReward(config), 1.0)
    
    reward1 = composite.compute_reward(state, action, next_state)
    
    # Increase adherence weight
    composite.set_component_weight('adherence', 5.0)
    reward2 = composite.compute_reward(state, action, next_state)
    
    # Reward should change
    assert reward2 != reward1, "Changing weights should affect total reward"


# ========== Reward Shaping Tests ==========

def test_normalize_reward():
    """Test reward normalization."""
    reward = 50.0
    normalized = normalize_reward(reward, min_val=0.0, max_val=100.0)
    
    assert 0.0 <= normalized <= 1.0, "Normalized reward should be in [0, 1]"
    assert abs(normalized - 0.5) < 0.01, f"Expected ~0.5, got {normalized}"


def test_clip_reward():
    """Test reward clipping."""
    # Test upper bound
    clipped_high = clip_reward(100.0, clip_range=(-10.0, 10.0))
    assert clipped_high == 10.0
    
    # Test lower bound
    clipped_low = clip_reward(-100.0, clip_range=(-10.0, 10.0))
    assert clipped_low == -10.0
    
    # Test within range
    clipped_mid = clip_reward(5.0, clip_range=(-10.0, 10.0))
    assert clipped_mid == 5.0


def test_potential_based_shaping():
    """Test potential-based reward shaping."""
    state = {
        'glucose': 150.0,
        'systolic_bp': 130.0,
        'adherence_score': 0.6
    }
    
    next_state = {
        'glucose': 120.0,  # Improved
        'systolic_bp': 115.0,  # Improved
        'adherence_score': 0.7  # Improved
    }
    
    shaping = potential_based_shaping(
        state,
        next_state,
        health_potential_function,
        gamma=0.99
    )
    
    # Should be positive (moving to better state)
    assert shaping > 0, "Shaping should be positive for improvement"


# ========== Clinical Scenario Tests ==========

def test_diabetes_management_success():
    """Test full diabetes management success scenario."""
    config = RewardConfig()
    
    # Patient with diabetes, starting treatment
    state = {
        'glucose': 160.0,
        'hba1c': 8.0,
        'adherence_score': 0.5,
        'adherence_streak_days': 0
    }
    
    # Good treatment: appropriate insulin, reminders
    action = {
        'insulin_dose': 25.0,
        'num_reminders': 2,
        'schedule_appointment': True
    }
    
    # After 30 days: improvement
    next_state = {
        'glucose': 110.0,
        'hba1c': 7.0,
        'adherence_score': 0.85,
        'adherence_streak_days': 30
    }
    
    # Build composite reward
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    reward = composite.compute_reward(state, action, next_state)
    
    # Should be strongly positive
    assert reward > 1.0, "Successful treatment should give high reward"
    
    # Get breakdown
    breakdown = composite.get_reward_components(state, action, next_state)
    
    # Adherence should be positive
    assert breakdown['adherence_total'] > 0
    
    # Health should be positive
    assert breakdown['health_total'] > 0
    
    # Safety should be zero (no violations)
    assert breakdown['safety_total'] == 0.0
    
    # Cost should be small negative
    assert breakdown['cost_total'] < 0
    assert breakdown['cost_total'] > -1.0


def test_diabetes_management_failure():
    """Test diabetes management failure scenario."""
    config = RewardConfig()
    
    # Patient not adhering to treatment
    state = {
        'glucose': 180.0,
        'adherence_score': 0.7,
        'systolic_bp': 140.0
    }
    
    # Inadequate action
    action = {
        'insulin_dose': 10.0,  # Too low
        'num_reminders': 0
    }
    
    # Condition worsens
    next_state = {
        'glucose': 250.0,
        'adherence_score': 0.4,
        'systolic_bp': 155.0,
        'hospitalized': True,
        'hospital_days': 3
    }
    
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    reward = composite.compute_reward(state, action, next_state)
    
    # Should be strongly negative
    assert reward < -2.0, "Treatment failure should give strong negative reward"


# ========== Run Tests ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
