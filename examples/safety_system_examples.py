"""
Example Usage of Safety System
Demonstrates how to integrate safety system with your RL agent
"""

import numpy as np
import torch
import sys
sys.path.append('..')

from models.safety import (
    SafetyConfig,
    SafetyLayer,
    SafeRLAgent,
    SafetyCritic,
    train_safety_critic,
    generate_safety_report,
    print_safety_report
)


def example_basic_usage():
    """Example 1: Basic safety checking"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Safety Checking")
    print("=" * 70)
    
    # Step 1: Create configuration
    config = SafetyConfig()
    
    # Step 2: Initialize safety layer
    safety_layer = SafetyLayer(config)
    
    # Step 3: Define patient state and action
    patient_state = {
        'glucose': 180.0,
        'blood_pressure_systolic': 145.0,
        'blood_pressure_diastolic': 90.0,
        'age': 55,
        'allergies': [],
        'current_medications': [],
        'conditions': []
    }
    
    proposed_action = {
        'medication_type': 'insulin',
        'dosage': 25.0,
        'next_appointment_days': 14,
        'reminder_frequency': 3
    }
    
    # Step 4: Check if action is safe
    is_safe, violations = safety_layer.check_action_safety(patient_state, proposed_action)
    
    print(f"\nPatient State: {patient_state}")
    print(f"Proposed Action: {proposed_action}")
    print(f"\nIs Safe: {is_safe}")
    
    if not is_safe:
        print(f"Violations: {violations}")
    else:
        print("Action is safe to execute!")


def example_safety_enforcement():
    """Example 2: Enforcing safety on unsafe actions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Safety Enforcement")
    print("=" * 70)
    
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    
    patient_state = {
        'glucose': 85.0,  # Lower glucose
        'age': 55,
        'allergies': [],
        'current_medications': []
    }
    
    # Propose an unsafe action (too much insulin for current glucose)
    unsafe_action_array = np.array([0.9, 0.5, 0.4, 0.0, 0.0])  # High dosage
    
    print(f"\nPatient glucose: {patient_state['glucose']} mg/dL")
    print(f"Proposed action (unsafe): {unsafe_action_array}")
    
    # Enforce safety
    safe_action = safety_layer.enforce_safety(patient_state, unsafe_action_array)
    
    print(f"Safe action (corrected): {safe_action}")
    print("\nSafety enforcement successful!")


def example_rl_agent_integration():
    """Example 3: Integrating with RL agent"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: RL Agent Integration")
    print("=" * 70)
    
    # Mock RL agent (replace with your actual CQL agent)
    class MockCQLAgent:
        def select_action(self, state):
            # Return random action
            return np.random.rand(5)
    
    # Step 1: Create your base RL agent
    cql_agent = MockCQLAgent()
    
    # Step 2: Create safety layer
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    
    # Step 3: Wrap agent with safety
    safe_agent = SafeRLAgent(
        rl_agent=cql_agent,
        safety_layer=safety_layer,
        safety_threshold=0.8
    )
    
    # Step 4: Use safe agent instead of base agent
    patient_state = {
        'glucose': 150,
        'blood_pressure_systolic': 120,
        'age': 45,
        'allergies': [],
        'current_medications': []
    }
    
    print("\nSelecting actions with safety enforcement:")
    for i in range(5):
        action = safe_agent.select_action(patient_state)
        print(f"  Action {i+1}: {action}")
    
    # Step 5: Check statistics
    stats = safe_agent.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Safety overrides: {stats['total_overrides']}")
    print(f"  Override rate: {stats['override_rate']:.2%}")


def example_safety_critic_training():
    """Example 4: Training safety critic"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Training Safety Critic")
    print("=" * 70)
    
    state_dim = 10
    action_dim = 5
    
    # Step 1: Create safety critic
    critic = SafetyCritic(state_dim, action_dim, hidden_dim=256)
    
    # Step 2: Prepare training data
    # In practice, collect these from your environment
    print("\nGenerating training data...")
    
    # Safe examples: actions that led to safe states
    safe_dataset = [
        (np.random.randn(state_dim), np.random.randn(action_dim) * 0.5)
        for _ in range(500)
    ]
    
    # Unsafe examples: actions that led to unsafe states
    unsafe_dataset = [
        (np.random.randn(state_dim), np.random.randn(action_dim) * 2.0)
        for _ in range(500)
    ]
    
    # Step 3: Train critic
    print("Training safety critic...")
    history = train_safety_critic(
        critic,
        safe_dataset,
        unsafe_dataset,
        num_epochs=50,
        batch_size=32,
        lr=1e-4,
        device='cpu'
    )
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {history['losses'][-1]:.4f}")
    
    if history['accuracies']:
        print(f"  Final accuracy: {history['accuracies'][-1]:.4f}")
    
    # Step 4: Integrate with safety layer
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    safety_layer.set_safety_critic(critic)
    
    print("\nSafety critic integrated with safety layer!")


def example_safety_evaluation():
    """Example 5: Evaluating safety on trajectories"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Safety Evaluation")
    print("=" * 70)
    
    # Step 1: Create test trajectories
    print("\nGenerating test trajectories...")
    trajectories = []
    patient_ids = []
    
    for i in range(20):
        trajectory = []
        for t in range(30):
            state = {
                'glucose': 100 + np.random.randn() * 30,
                'blood_pressure_systolic': 120 + np.random.randn() * 15,
                'blood_pressure_diastolic': 80 + np.random.randn() * 10,
                'heart_rate': 75 + np.random.randn() * 10,
                'action': {
                    'medication_type': 'insulin',
                    'dosage': 15 + np.random.rand() * 20
                }
            }
            trajectory.append(state)
        
        trajectories.append(trajectory)
        patient_ids.append(f"patient_{i:03d}")
    
    # Step 2: Set up safety system
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    
    # Step 3: Generate comprehensive safety report
    print("\nGenerating safety report...")
    report = generate_safety_report(
        trajectories=trajectories,
        constraints=safety_layer.constraints,
        safe_ranges=config.safe_ranges,
        patient_ids=patient_ids
    )
    
    # Step 4: Display report
    print_safety_report(report, verbose=True)
    
    # Step 5: Access specific metrics
    print("\nKey Metrics:")
    print(f"  Safety Index: {report['overall_metrics']['safety_index']:.4f}")
    print(f"  Violation Severity: {report['overall_metrics']['violation_severity']:.4f}")
    
    print("\nConstraint Satisfaction Rates:")
    for constraint_name, rate in report['constraint_metrics']['satisfaction_rates'].items():
        print(f"  {constraint_name}: {rate:.2%}")


def example_custom_configuration():
    """Example 6: Custom safety configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Safety Configuration")
    print("=" * 70)
    
    # Create custom configuration
    custom_config = SafetyConfig(
        # Custom drug limits
        drug_limits={
            'insulin': (0.0, 50.0),  # Stricter insulin limit
            'metformin': (500.0, 1500.0),  # Different metformin range
            'custom_drug': (10.0, 100.0)  # Add new drug
        },
        
        # Custom physiological ranges
        safe_ranges={
            'glucose': (80.0, 180.0),  # Tighter glucose range
            'blood_pressure_systolic': (100.0, 130.0),  # Tighter BP range
        },
        
        # Custom safety threshold
        safety_threshold=0.9,  # Higher threshold for safety critic
        
        # Custom frequency limits
        max_reminders_per_week=5,  # Fewer reminders
        min_appointment_interval_days=7  # Longer interval
    )
    
    # Use custom configuration
    safety_layer = SafetyLayer(custom_config)
    
    print("Custom configuration created:")
    print(f"  Drug limits: {custom_config.drug_limits}")
    print(f"  Safe ranges: {custom_config.safe_ranges}")
    print(f"  Safety threshold: {custom_config.safety_threshold}")
    
    # Test with custom config
    state = {'glucose': 175, 'age': 45}
    action = {'medication_type': 'insulin', 'dosage': 45}
    
    is_safe, violations = safety_layer.check_action_safety(state, action)
    print(f"\nTesting action with custom config:")
    print(f"  Glucose: {state['glucose']} (safe range: {custom_config.safe_ranges['glucose']})")
    print(f"  Insulin dose: {action['dosage']} (safe range: {custom_config.drug_limits['insulin']})")
    print(f"  Is safe: {is_safe}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("SAFETY SYSTEM USAGE EXAMPLES")
    print("=" * 70)
    
    example_basic_usage()
    example_safety_enforcement()
    example_rl_agent_integration()
    example_safety_critic_training()
    example_safety_evaluation()
    example_custom_configuration()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
