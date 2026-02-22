"""
Example Usage of Reward Functions

This script demonstrates how to use the reward functions for healthcare RL.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

# Import reward components
from src.rewards import (
    RewardConfig,
    CompositeRewardFunction,
    AdherenceReward,
    HealthOutcomeReward,
    SafetyPenalty,
    CostEffectivenessReward,
    ConservativeRewardConfig,
    AggressiveRewardConfig
)


def create_sample_states() -> List[Tuple[Dict, Dict, Dict]]:
    """
    Create sample patient scenarios for testing.
    
    Returns:
        List of (state, action, next_state) tuples
    """
    scenarios = []
    
    # Scenario 1: Good diabetes management
    scenarios.append((
        # State
        {
            'glucose': 140.0,
            'systolic_bp': 125.0,
            'diastolic_bp': 78.0,
            'adherence_score': 0.7,
            'adherence_streak_days': 20
        },
        # Action
        {
            'insulin_dose': 22.0,
            'num_reminders': 2,
            'schedule_appointment': True
        },
        # Next State
        {
            'glucose': 115.0,
            'systolic_bp': 118.0,
            'diastolic_bp': 75.0,
            'adherence_score': 0.85,
            'adherence_streak_days': 21
        }
    ))
    
    # Scenario 2: Hypoglycemia event
    scenarios.append((
        # State
        {
            'glucose': 80.0,
            'adherence_score': 0.9
        },
        # Action
        {
            'insulin_dose': 30.0  # Too much
        },
        # Next State
        {
            'glucose': 50.0,  # Severe hypoglycemia
            'adherence_score': 0.9
        }
    ))
    
    # Scenario 3: Poor adherence leading to complications
    scenarios.append((
        # State
        {
            'glucose': 180.0,
            'adherence_score': 0.4,
            'systolic_bp': 140.0
        },
        # Action
        {
            'insulin_dose': 15.0,
            'num_reminders': 0
        },
        # Next State
        {
            'glucose': 250.0,
            'adherence_score': 0.3,
            'systolic_bp': 155.0,
            'emergency_visit': True
        }
    ))
    
    return scenarios


def demo_individual_rewards():
    """Demonstrate individual reward components."""
    print("=" * 60)
    print("INDIVIDUAL REWARD COMPONENTS DEMO")
    print("=" * 60)
    
    config = RewardConfig()
    scenarios = create_sample_states()
    
    # Test each reward component
    adherence_reward = AdherenceReward(config)
    health_reward = HealthOutcomeReward(config)
    safety_penalty = SafetyPenalty(config)
    cost_reward = CostEffectivenessReward(config)
    
    for i, (state, action, next_state) in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"  State: glucose={state.get('glucose', 'N/A')}, "
              f"adherence={state.get('adherence_score', 'N/A')}")
        
        # Adherence
        adh_r = adherence_reward.compute_reward(state, action, next_state)
        print(f"  Adherence Reward: {adh_r:.3f}")
        
        # Health
        health_r = health_reward.compute_reward(state, action, next_state)
        print(f"  Health Reward: {health_r:.3f}")
        
        # Safety
        safety_r = safety_penalty.compute_reward(state, action, next_state)
        print(f"  Safety Penalty: {safety_r:.3f}")
        
        # Cost
        cost_r = cost_reward.compute_reward(state, action, next_state)
        print(f"  Cost Penalty: {cost_r:.3f}")


def demo_composite_reward():
    """Demonstrate composite reward function."""
    print("\n" + "=" * 60)
    print("COMPOSITE REWARD DEMO")
    print("=" * 60)
    
    config = RewardConfig()
    scenarios = create_sample_states()
    
    # Create composite reward
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    print(f"\nComponent Weights:")
    weights = composite.get_component_weights()
    for name, weight in weights.items():
        print(f"  {name}: {weight}")
    
    for i, (state, action, next_state) in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        
        # Compute total reward
        total_reward = composite.compute_reward(state, action, next_state)
        print(f"Total Reward: {total_reward:.3f}")
        
        # Get detailed breakdown
        breakdown = composite.get_reward_components(state, action, next_state)
        
        print("Detailed Breakdown:")
        for component in ['adherence', 'health', 'safety', 'cost']:
            weighted_key = f'{component}_weighted'
            if weighted_key in breakdown:
                print(f"  {component.capitalize()}: {breakdown[weighted_key]:.3f}")


def demo_different_configs():
    """Demonstrate different reward configurations."""
    print("\n" + "=" * 60)
    print("DIFFERENT CONFIGURATIONS DEMO")
    print("=" * 60)
    
    # Get a challenging scenario (hypoglycemia)
    scenarios = create_sample_states()
    state, action, next_state = scenarios[1]  # Hypoglycemia scenario
    
    print("\nScenario: Hypoglycemia Event")
    print(f"  Glucose: {state['glucose']} → {next_state['glucose']}")
    
    # Test with different configs
    configs = {
        'Standard': RewardConfig(),
        'Conservative (Safety-First)': ConservativeRewardConfig(),
        'Aggressive (Health-First)': AggressiveRewardConfig()
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name} Configuration:")
        print(f"  Safety weight: {config.w_safety}")
        print(f"  Health weight: {config.w_health}")
        
        # Create composite reward
        composite = CompositeRewardFunction(config)
        composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
        composite.add_component('health', HealthOutcomeReward(config), config.w_health)
        composite.add_component('safety', SafetyPenalty(config), config.w_safety)
        composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
        
        total_reward = composite.compute_reward(state, action, next_state)
        print(f"  Total Reward: {total_reward:.3f}")
        
        breakdown = composite.get_reward_components(state, action, next_state)
        print(f"  Safety Component: {breakdown.get('safety_weighted', 0):.3f}")


def demo_reward_breakdown():
    """Demonstrate detailed reward breakdown for analysis."""
    print("\n" + "=" * 60)
    print("DETAILED REWARD BREAKDOWN DEMO")
    print("=" * 60)
    
    config = RewardConfig()
    
    # Good scenario
    state = {
        'glucose': 150.0,
        'systolic_bp': 130.0,
        'adherence_score': 0.6,
        'adherence_streak_days': 15
    }
    
    action = {
        'insulin_dose': 20.0,
        'num_reminders': 2
    }
    
    next_state = {
        'glucose': 110.0,
        'systolic_bp': 115.0,
        'adherence_score': 0.85,
        'adherence_streak_days': 16
    }
    
    # Create composite reward
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    # Get full breakdown
    breakdown = composite.get_reward_components(state, action, next_state)
    
    print("\nFull Reward Breakdown:")
    print("-" * 60)
    
    # Adherence components
    print("\nAdherence Components:")
    for key, value in breakdown.items():
        if key.startswith('adherence_') and not key.endswith('_weighted'):
            print(f"  {key}: {value:.3f}")
    
    # Health components
    print("\nHealth Components:")
    for key, value in breakdown.items():
        if key.startswith('health_') and not key.endswith('_weighted'):
            print(f"  {key}: {value:.3f}")
    
    # Safety components
    print("\nSafety Components:")
    for key, value in breakdown.items():
        if key.startswith('safety_') and not key.endswith('_weighted'):
            print(f"  {key}: {value:.3f}")
    
    # Cost components
    print("\nCost Components:")
    for key, value in breakdown.items():
        if key.startswith('cost_') and not key.endswith('_weighted'):
            print(f"  {key}: {value:.3f}")
    
    # Weighted totals
    print("\nWeighted Totals:")
    for component in ['adherence', 'health', 'safety', 'cost']:
        key = f'{component}_weighted'
        if key in breakdown:
            print(f"  {component.capitalize()}: {breakdown[key]:.3f}")
    
    print(f"\nFinal Total: {breakdown['composite_total']:.3f}")


def demo_episode_simulation():
    """Simulate a full episode with reward tracking."""
    print("\n" + "=" * 60)
    print("EPISODE SIMULATION DEMO")
    print("=" * 60)
    
    config = RewardConfig()
    
    # Create composite reward
    composite = CompositeRewardFunction(config)
    composite.add_component('adherence', AdherenceReward(config), config.w_adherence)
    composite.add_component('health', HealthOutcomeReward(config), config.w_health)
    composite.add_component('safety', SafetyPenalty(config), config.w_safety)
    composite.add_component('cost', CostEffectivenessReward(config), config.w_cost)
    
    # Simulate a 10-step episode
    print("\nSimulating 10-step treatment episode...")
    
    # Initial state
    state = {
        'glucose': 180.0,
        'systolic_bp': 135.0,
        'adherence_score': 0.5,
        'adherence_streak_days': 0
    }
    
    total_reward = 0.0
    rewards_history = []
    
    for step in range(10):
        # Simulate action (simplified)
        action = {
            'insulin_dose': 20.0 + np.random.randn() * 2,
            'num_reminders': 2 if step % 3 == 0 else 1
        }
        
        # Simulate next state (simplified dynamics)
        next_state = {
            'glucose': max(60, state['glucose'] - 8 + np.random.randn() * 5),
            'systolic_bp': max(90, state['systolic_bp'] - 2 + np.random.randn() * 3),
            'adherence_score': min(1.0, state['adherence_score'] + 0.05),
            'adherence_streak_days': state['adherence_streak_days'] + 1
        }
        
        # Compute reward
        reward = composite.compute_reward(state, action, next_state)
        total_reward += reward
        rewards_history.append(reward)
        
        print(f"  Step {step + 1}: Glucose {next_state['glucose']:.1f}, "
              f"Adherence {next_state['adherence_score']:.2f}, "
              f"Reward {reward:.3f}")
        
        # Update state
        state = next_state
    
    print(f"\nEpisode Summary:")
    print(f"  Total Reward: {total_reward:.3f}")
    print(f"  Average Reward: {np.mean(rewards_history):.3f}")
    print(f"  Final Glucose: {state['glucose']:.1f}")
    print(f"  Final Adherence: {state['adherence_score']:.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("HEALTHCARE RL REWARD FUNCTIONS - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Run each demo
    demo_individual_rewards()
    demo_composite_reward()
    demo_different_configs()
    demo_reward_breakdown()
    demo_episode_simulation()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
