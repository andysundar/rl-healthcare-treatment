"""
Tests for Safety System
Comprehensive tests for all safety components
"""

import numpy as np
import torch
import sys
sys.path.append('..')

from src.models.safety import (
    SafetyConfig,
    SafetyLayer,
    SafeRLAgent,
    DosageConstraint,
    PhysiologicalConstraint,
    ContraindicationConstraint,
    FrequencyConstraint,
    SafetyCritic,
    train_safety_critic,
    safety_index,
    violation_rate,
    generate_safety_report,
    print_safety_report
)


def test_dosage_constraint():
    """Test dosage constraint"""
    print("\n" + "=" * 60)
    print("TEST 1: Dosage Constraint")
    print("=" * 60)
    
    drug_limits = {'insulin': (0.0, 100.0), 'metformin': (500.0, 2000.0)}
    constraint = DosageConstraint(drug_limits)
    
    # Test 1: Safe dosage
    state = {'age': 45}
    action = {'medication_type': 'insulin', 'dosage': 50.0}
    is_safe, msg = constraint.check(state, action)
    assert is_safe, "Safe dosage should pass"
    print("✓ Safe dosage test passed")
    
    # Test 2: Dosage too high
    action = {'medication_type': 'insulin', 'dosage': 150.0}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Too high dosage should fail"
    print("✓ High dosage test passed")
    
    # Test 3: Dosage too low
    action = {'medication_type': 'insulin', 'dosage': -10.0}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Negative dosage should fail"
    print("✓ Low dosage test passed")
    
    # Test 4: Age-adjusted limits
    state = {'age': 70}  # Elderly patient
    action = {'medication_type': 'insulin', 'dosage': 80.0}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Dosage should be adjusted for elderly"
    print("✓ Age-adjusted test passed")
    
    print("All dosage constraint tests passed!")


def test_physiological_constraint():
    """Test physiological constraint"""
    print("\n" + "=" * 60)
    print("TEST 2: Physiological Constraint")
    print("=" * 60)
    
    safe_ranges = {
        'glucose': (70.0, 200.0),
        'blood_pressure_systolic': (90.0, 140.0)
    }
    constraint = PhysiologicalConstraint(safe_ranges)
    
    # Test 1: Safe action
    state = {'glucose': 150.0, 'blood_pressure_systolic': 120.0}
    action = {'medication_type': 'insulin', 'dosage': 5.0}
    is_safe, msg = constraint.check(state, action)
    assert is_safe, "Safe action should pass"
    print("✓ Safe physiological action test passed")
    
    # Test 2: Action that would cause hypoglycemia
    state = {'glucose': 80.0}
    action = {'medication_type': 'insulin', 'dosage': 10.0}  # Would drop to ~-220
    is_safe, msg = constraint.check(state, action)
    # This should be unsafe due to predicted hypoglycemia
    print(f"  Hypoglycemia test - Safe: {is_safe}, Message: {msg}")
    print("✓ Hypoglycemia prediction test completed")


def test_contraindication_constraint():
    """Test contraindication constraint"""
    print("\n" + "=" * 60)
    print("TEST 3: Contraindication Constraint")
    print("=" * 60)
    
    constraint = ContraindicationConstraint()
    
    # Test 1: Drug allergy
    state = {'allergies': ['insulin'], 'current_medications': [], 'age': 45}
    action = {'medication_type': 'insulin'}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Should detect allergy"
    print("✓ Allergy detection test passed")
    
    # Test 2: Drug interaction
    state = {'allergies': [], 'current_medications': ['sulfonamides'], 'age': 45}
    action = {'medication_type': 'insulin'}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Should detect interaction"
    print("✓ Drug interaction test passed")
    
    # Test 3: Age restriction
    state = {'allergies': [], 'current_medications': [], 'age': 12}
    action = {'medication_type': 'aspirin'}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Should enforce age restriction"
    print("✓ Age restriction test passed")
    
    print("All contraindication constraint tests passed!")


def test_frequency_constraint():
    """Test frequency constraint"""
    print("\n" + "=" * 60)
    print("TEST 4: Frequency Constraint")
    print("=" * 60)
    
    constraint = FrequencyConstraint(max_reminders_per_week=7, min_appointment_interval_days=3)
    
    # Test 1: Valid reminder frequency
    state = {}
    action = {'reminder_frequency': 3}
    is_safe, msg = constraint.check(state, action)
    assert is_safe, "Valid frequency should pass"
    print("✓ Valid frequency test passed")
    
    # Test 2: Too many reminders
    action = {'reminder_frequency': 10}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Too many reminders should fail"
    print("✓ Excessive reminders test passed")
    
    # Test 3: Appointment too soon
    action = {'next_appointment_days': 1}
    is_safe, msg = constraint.check(state, action)
    assert not is_safe, "Appointment too soon should fail"
    print("✓ Appointment interval test passed")
    
    print("All frequency constraint tests passed!")


def test_safety_layer():
    """Test SafetyLayer integration"""
    print("\n" + "=" * 60)
    print("TEST 5: Safety Layer Integration")
    print("=" * 60)
    
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    
    # Test 1: Check safe action
    state = {'glucose': 150, 'age': 45, 'allergies': [], 'current_medications': []}
    action = {
        'medication_type': 'insulin',
        'dosage': 20.0,
        'next_appointment_days': 14,
        'reminder_frequency': 3
    }
    is_safe, violations = safety_layer.check_action_safety(state, action)
    assert is_safe, "Safe action should pass all constraints"
    print("✓ Safe action test passed")
    
    # Test 2: Check unsafe action (allergy)
    state = {'glucose': 150, 'age': 45, 'allergies': ['insulin'], 'current_medications': []}
    action = {'medication_type': 'insulin', 'dosage': 20.0}
    is_safe, violations = safety_layer.check_action_safety(state, action)
    assert not is_safe, "Should detect allergy"
    assert 'ContraindicationConstraint' in violations
    print("✓ Unsafe action detection test passed")
    
    # Test 3: Enforce safety
    proposed_action = np.array([0.9, 0.8, 0.7, 0.0, 0.0])
    safe_action = safety_layer.enforce_safety(state, proposed_action)
    assert safe_action is not None, "Should return safe action"
    print("✓ Safety enforcement test passed")
    
    # Test 4: Get violation statistics
    stats = safety_layer.get_violation_statistics()
    assert 'total_violations' in stats
    print(f"✓ Violation statistics: {stats}")
    
    print("All safety layer tests passed!")


def test_safety_critic():
    """Test SafetyCritic neural network"""
    print("\n" + "=" * 60)
    print("TEST 6: Safety Critic")
    print("=" * 60)
    
    state_dim = 10
    action_dim = 3
    critic = SafetyCritic(state_dim, action_dim, hidden_dim=64)
    
    # Test 1: Forward pass
    state = torch.randn(5, state_dim)
    action = torch.randn(5, action_dim)
    output = critic.forward(state, action)
    assert output.shape == (5, 1), "Output shape should be (batch_size, 1)"
    assert torch.all((output >= 0) & (output <= 1)), "Output should be in [0, 1]"
    print("✓ Forward pass test passed")
    
    # Test 2: Training
    # Create dummy dataset
    safe_dataset = [(np.random.randn(state_dim), np.random.randn(action_dim)) for _ in range(100)]
    unsafe_dataset = [(np.random.randn(state_dim), np.random.randn(action_dim)) for _ in range(100)]
    
    history = train_safety_critic(
        critic, 
        safe_dataset, 
        unsafe_dataset,
        num_epochs=10,
        batch_size=32,
        lr=1e-3
    )
    
    assert 'losses' in history
    assert len(history['losses']) > 0
    print(f"✓ Training test passed - Final loss: {history['losses'][-1]:.4f}")
    
    # Test 3: Prediction
    test_state = torch.randn(1, state_dim)
    test_action = torch.randn(1, action_dim)
    is_safe, score = critic.predict_safety(test_state, test_action, threshold=0.5)
    assert isinstance(is_safe, bool)
    assert 0 <= score <= 1
    print(f"✓ Prediction test passed - Safety score: {score:.4f}")
    
    print("All safety critic tests passed!")


def test_safe_rl_agent():
    """Test SafeRLAgent wrapper"""
    print("\n" + "=" * 60)
    print("TEST 7: SafeRLAgent Wrapper")
    print("=" * 60)
    
    # Mock RL agent
    class MockRLAgent:
        def select_action(self, state):
            return np.random.rand(5)
    
    base_agent = MockRLAgent()
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    safe_agent = SafeRLAgent(base_agent, safety_layer)
    
    # Test action selection
    state = {
        'glucose': 150,
        'age': 45,
        'allergies': [],
        'current_medications': [],
        'blood_pressure_systolic': 120
    }
    
    for _ in range(10):
        action = safe_agent.select_action(state)
        assert action is not None, "Should return action"
        assert isinstance(action, np.ndarray), "Should return numpy array"
    
    # Check statistics
    stats = safe_agent.get_statistics()
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Total overrides: {stats['total_overrides']}")
    print(f"  Override rate: {stats['override_rate']:.4f}")
    
    assert stats['total_actions'] == 10
    print("✓ SafeRLAgent test passed")


def test_safety_metrics():
    """Test safety metrics computation"""
    print("\n" + "=" * 60)
    print("TEST 8: Safety Metrics")
    print("=" * 60)
    
    # Create dummy trajectories
    trajectories = []
    for i in range(10):
        trajectory = []
        for t in range(20):
            state = {
                'glucose': 100 + np.random.randn() * 20,
                'blood_pressure_systolic': 120 + np.random.randn() * 10,
                'action': {'medication_type': 'insulin', 'dosage': 10 + np.random.rand() * 20}
            }
            trajectory.append(state)
        trajectories.append(trajectory)
    
    safe_ranges = {
        'glucose': (70.0, 200.0),
        'blood_pressure_systolic': (90.0, 140.0)
    }
    
    # Test safety index
    si = safety_index(trajectories, safe_ranges)
    print(f"  Safety Index: {si:.4f}")
    assert 0 <= si <= 1, "Safety index should be in [0, 1]"
    print("✓ Safety index test passed")
    
    # Test comprehensive report
    config = SafetyConfig()
    safety_layer = SafetyLayer(config)
    
    report = generate_safety_report(
        trajectories,
        safety_layer.constraints,
        safe_ranges
    )
    
    assert 'overall_metrics' in report
    assert 'constraint_metrics' in report
    print("✓ Safety report generation test passed")
    
    # Print report
    print("\nSample Safety Report:")
    print_safety_report(report, verbose=False)


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING ALL SAFETY SYSTEM TESTS")
    print("=" * 60)
    
    try:
        test_dosage_constraint()
        test_physiological_constraint()
        test_contraindication_constraint()
        test_frequency_constraint()
        test_safety_layer()
        test_safety_critic()
        test_safe_rl_agent()
        test_safety_metrics()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
