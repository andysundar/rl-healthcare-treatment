"""
Example usage of baseline policies for healthcare RL.

This script demonstrates:
1. Creating and configuring different baseline policies
2. Training policies on historical data
3. Evaluating and comparing baselines
4. Generating comparison reports
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baselines import (
    create_diabetes_rule_policy,
    create_hypertension_rule_policy,
    create_random_policy,
    create_safe_random_policy,
    create_behavior_cloning_policy,
    create_mean_action_policy,
    create_regression_policy,
    create_knn_policy,
    compare_all_baselines,
    BaselineComparator
)


def generate_synthetic_diabetes_data(n_patients: int = 1000, 
                                    trajectory_length: int = 30):
    """
    Generate synthetic diabetes management data.
    
    State: [glucose, insulin_level, hba1c, age, bmi, ...]
    Action: [insulin_dosage]
    Reward: Based on glucose control
    """
    np.random.seed(42)
    
    state_dim = 10
    action_dim = 1
    
    all_data = []
    
    for patient_id in range(n_patients):
        # Patient-specific parameters
        baseline_glucose = np.random.uniform(100, 150)
        insulin_sensitivity = np.random.uniform(0.5, 1.5)
        
        # Initialize patient state
        state = np.random.randn(state_dim)
        state[0] = baseline_glucose  # Glucose
        state[1] = 0.5  # Insulin level
        state[2] = np.random.uniform(5.5, 7.0)  # HbA1c
        
        for t in range(trajectory_length):
            # Historical action (expert policy with noise)
            if state[0] > 180:
                action = np.array([min(1.0, 0.7 + np.random.randn() * 0.1)])
            elif state[0] < 80:
                action = np.array([max(0.0, 0.3 + np.random.randn() * 0.1)])
            else:
                action = np.array([np.clip(0.5 + np.random.randn() * 0.1, 0, 1)])
            
            # Compute reward
            target_glucose = 120
            glucose_error = abs(state[0] - target_glucose)
            reward = -glucose_error / 100.0  # Negative error
            
            # Adverse events
            if state[0] < 70 or state[0] > 250:
                reward -= 5.0  # Safety violation penalty
            
            # Next state simulation (simple dynamics)
            next_state = state.copy()
            
            # Glucose dynamics (simplified)
            insulin_effect = action[0] * insulin_sensitivity * 50
            next_state[0] = state[0] - insulin_effect + np.random.randn() * 10
            next_state[0] = np.clip(next_state[0], 50, 350)
            
            # Update insulin level
            next_state[1] = action[0]
            
            # Episode done
            done = (t == trajectory_length - 1)
            
            # Store transition
            all_data.append((state.copy(), action, reward, next_state.copy(), done))
            
            # Update state
            state = next_state
    
    return all_data, state_dim, action_dim


def example_rule_based_policy():
    """Example 1: Rule-based policy."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Rule-Based Policy")
    print("="*60 + "\n")
    
    # Create diabetes rule-based policy
    policy = create_diabetes_rule_policy(state_dim=10, action_dim=1)
    
    print(f"Created policy: {policy}")
    print(f"Number of rules: {len(policy.rules)}")
    
    # Test on sample states
    test_states = [
        np.array([250, 0.5, 7.0, 50, 28] + [0]*5),  # High glucose
        np.array([60, 0.5, 7.0, 50, 28] + [0]*5),   # Low glucose
        np.array([120, 0.5, 6.5, 50, 28] + [0]*5),  # Normal glucose
    ]
    
    print("\nTesting rule applications:")
    for i, state in enumerate(test_states):
        action = policy.select_action(state)
        applicable_rules = policy.get_applicable_rules(state)
        
        print(f"\nState {i+1}: Glucose = {state[0]:.1f}")
        print(f"  Action: {action[0]:.3f}")
        print(f"  Applicable rules: {[r.name for r in applicable_rules]}")
    
    # Get usage statistics
    stats = policy.get_rule_statistics()
    print("\nRule usage statistics:")
    for rule_name, rule_stats in stats['rule_usage'].items():
        print(f"  {rule_name}: {rule_stats['count']} times ({rule_stats['rate']:.2%})")


def example_behavior_cloning():
    """Example 2: Behavior cloning from historical data."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Behavior Cloning")
    print("="*60 + "\n")
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    data, state_dim, action_dim = generate_synthetic_diabetes_data(
        n_patients=500, 
        trajectory_length=20
    )
    
    # Split into train/val
    n_train = int(0.8 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    # Extract states and actions
    train_states = np.array([d[0] for d in train_data])
    train_actions = np.array([d[1] for d in train_data])
    val_states = np.array([d[0] for d in val_data])
    val_actions = np.array([d[1] for d in val_data])
    
    print(f"Training samples: {len(train_states)}")
    print(f"Validation samples: {len(val_states)}")
    
    # Create and train behavior cloning policy
    policy = create_behavior_cloning_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        learning_rate=1e-3
    )
    
    print(f"\nCreated policy: {policy}")
    print("Training...")
    
    history = policy.train(
        states=train_states,
        actions=train_actions,
        val_states=val_states,
        val_actions=val_actions,
        epochs=50,
        batch_size=256,
        verbose=True
    )
    
    print(f"\nFinal training loss: {history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Test predictions
    test_state = val_states[0]
    predicted_action = policy.select_action(test_state)
    true_action = val_actions[0]
    
    print(f"\nExample prediction:")
    print(f"  True action: {true_action[0]:.3f}")
    print(f"  Predicted action: {predicted_action[0]:.3f}")
    print(f"  Error: {abs(true_action[0] - predicted_action[0]):.3f}")


def example_statistical_baselines():
    """Example 3: Statistical baseline policies."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Statistical Baselines")
    print("="*60 + "\n")
    
    # Generate data
    data, state_dim, action_dim = generate_synthetic_diabetes_data(
        n_patients=200,
        trajectory_length=20
    )
    
    train_states = np.array([d[0] for d in data])
    train_actions = np.array([d[1] for d in data])
    
    # 1. Mean action policy
    print("1. Mean Action Policy")
    mean_policy = create_mean_action_policy(action_dim=action_dim, state_dim=state_dim)
    mean_policy.fit(train_states, train_actions)
    print(f"   Mean action: {mean_policy.mean_action[0]:.3f}")
    
    # 2. Ridge regression policy
    print("\n2. Ridge Regression Policy")
    ridge_policy = create_regression_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        regression_type='ridge',
        alpha=1.0
    )
    ridge_policy.fit(train_states, train_actions)
    print("   Training complete")
    
    # 3. KNN policy
    print("\n3. K-Nearest Neighbors Policy")
    knn_policy = create_knn_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        k=5,
        metric='euclidean'
    )
    knn_policy.fit(train_states, train_actions)
    print(f"   Fitted with k={knn_policy.k}")
    
    # Compare predictions
    test_state = train_states[0]
    
    print("\nComparison on sample state:")
    print(f"  Mean Action:  {mean_policy.select_action(test_state)[0]:.3f}")
    print(f"  Ridge Regression: {ridge_policy.select_action(test_state)[0]:.3f}")
    print(f"  KNN: {knn_policy.select_action(test_state)[0]:.3f}")
    print(f"  True Action: {train_actions[0][0]:.3f}")


def example_comprehensive_comparison():
    """Example 4: Comprehensive baseline comparison."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Comprehensive Baseline Comparison")
    print("="*60 + "\n")
    
    # Generate data
    print("Generating synthetic data...")
    data, state_dim, action_dim = generate_synthetic_diabetes_data(
        n_patients=300,
        trajectory_length=25
    )
    
    # Split train/test
    n_train = int(0.7 * len(data))
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    train_states = np.array([d[0] for d in train_data])
    train_actions = np.array([d[1] for d in train_data])
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create all baseline policies
    print("\nCreating baseline policies...")
    
    # 1. Rule-based
    rule_policy = create_diabetes_rule_policy(state_dim=state_dim, action_dim=action_dim)
    
    # 2. Random (uniform)
    random_uniform = create_random_policy(
        action_dim=action_dim,
        state_dim=state_dim,
        seed=42,
        distribution='uniform'
    )
    
    # 3. Random (safe)
    random_safe = create_safe_random_policy(
        action_dim=action_dim,
        state_dim=state_dim,
        seed=42,
        num_samples=10
    )
    
    # 4. Mean action
    mean_policy = create_mean_action_policy(action_dim=action_dim, state_dim=state_dim)
    mean_policy.fit(train_states, train_actions)
    
    # 5. Ridge regression
    ridge_policy = create_regression_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        regression_type='ridge'
    )
    ridge_policy.fit(train_states, train_actions)
    
    # 6. KNN
    knn_policy = create_knn_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        k=5
    )
    knn_policy.fit(train_states, train_actions)
    
    # 7. Behavior cloning
    bc_policy = create_behavior_cloning_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64]
    )
    bc_policy.train(
        train_states,
        train_actions,
        epochs=20,
        batch_size=128,
        verbose=False
    )
    
    # Create baseline dictionary
    baselines = {
        'Rule-Based': rule_policy,
        'Random-Uniform': random_uniform,
        'Random-Safe': random_safe,
        'Mean-Action': mean_policy,
        'Ridge-Regression': ridge_policy,
        'KNN-5': knn_policy,
        'Behavior-Cloning': bc_policy
    }
    
    # Compare all baselines
    print("\nEvaluating all baselines...")
    results = compare_all_baselines(
        test_data=test_data,
        baselines_dict=baselines,
        output_path='baseline_comparison_report.md'
    )
    
    # Display results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60 + "\n")
    
    print(results.to_string())
    
    print("\nReport saved to: baseline_comparison_report.md")
    print("Raw results saved to: baseline_comparison_report.json")
    
    # Get best baseline
    comparator = BaselineComparator(test_data)
    for name, policy in baselines.items():
        comparator.add_baseline(name, policy)
    
    comparator.evaluate_all(verbose=False)
    best_name, best_policy, best_score = comparator.get_best_baseline('mean_reward')
    
    print(f"\nBest baseline: {best_name}")
    print(f"Mean reward: {best_score:.4f}")


def example_custom_metrics():
    """Example 5: Using custom evaluation metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Evaluation Metrics")
    print("="*60 + "\n")
    
    from src.models.baselines import (
        compute_action_stability,
        compute_safety_margin,
        compute_expected_return
    )
    
    # Generate data
    data, state_dim, action_dim = generate_synthetic_diabetes_data(
        n_patients=100,
        trajectory_length=20
    )
    
    # Create a baseline
    policy = create_diabetes_rule_policy(state_dim=state_dim, action_dim=action_dim)
    
    # Compute custom metrics
    print("Computing custom metrics...")
    
    stability = compute_action_stability(policy, data)
    safety_margin = compute_safety_margin(policy, data, safety_threshold=0.95)
    expected_return = compute_expected_return(policy, data, gamma=0.99)
    
    print(f"\nAction Stability (variance): {stability:.4f}")
    print(f"Safety Margin: {safety_margin:.4f}")
    print(f"Expected Return: {expected_return:.4f}")
    
    # Use in comparison
    custom_metrics = {
        'action_stability': compute_action_stability,
        'safety_margin': lambda p, d: compute_safety_margin(p, d, 0.95),
        'expected_return': lambda p, d: compute_expected_return(p, d, 0.99)
    }
    
    baselines = {
        'Rule-Based': policy,
        'Random': create_random_policy(action_dim=action_dim, state_dim=state_dim, seed=42)
    }
    
    results = compare_all_baselines(
        test_data=data,
        baselines_dict=baselines,
        custom_metrics=custom_metrics
    )
    
    print("\nComparison with custom metrics:")
    print(results[['mean_reward', 'action_stability', 'safety_margin', 'expected_return']].to_string())


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("BASELINE POLICIES - EXAMPLE USAGE")
    print("="*60)
    
    # Run examples
    example_rule_based_policy()
    example_behavior_cloning()
    example_statistical_baselines()
    example_comprehensive_comparison()
    example_custom_metrics()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
