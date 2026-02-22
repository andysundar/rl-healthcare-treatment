"""
Complete Evaluation Example

Demonstrates full evaluation pipeline for healthcare RL policies.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from evaluation import (
    EvaluationConfig,
    OffPolicyEvaluator,
    SafetyEvaluator,
    ClinicalEvaluator,
    PerformanceEvaluator,
    PolicyComparator,
    EvaluationVisualizer,
    Trajectory
)


def generate_synthetic_trajectories(n_trajectories=100):
    """Generate synthetic trajectories for demonstration."""
    trajectories = []
    
    for i in range(n_trajectories):
        T = np.random.randint(30, 90)  # Episode length
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Initialize state
        state = {
            'glucose': np.random.uniform(80, 150),
            'bp_systolic': np.random.uniform(100, 130),
            'adherence': np.random.uniform(0.5, 1.0)
        }
        
        for t in range(T):
            # Store state
            states.append(state.copy())
            
            # Random action
            action = {
                'medication_dosage': np.random.uniform(0, 1),
                'reminder_frequency': np.random.choice([0, 1, 2, 3])
            }
            actions.append(action)
            
            # Compute reward
            reward = 0.0
            if 80 <= state['glucose'] <= 130:
                reward += 1.0
            if state['adherence'] > 0.8:
                reward += 0.5
            rewards.append(reward)
            
            # Transition to next state
            next_state = {
                'glucose': state['glucose'] + np.random.randn() * 10,
                'bp_systolic': state['bp_systolic'] + np.random.randn() * 5,
                'adherence': min(1.0, state['adherence'] + action['reminder_frequency'] * 0.1)
            }
            next_states.append(next_state.copy())
            
            done = (t == T - 1)
            dones.append(done)
            
            state = next_state
        
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        trajectories.append(trajectory)
    
    return trajectories


def main():
    print("Healthcare RL Evaluation Example")
    print("="*70)
    
    # 1. Load configuration
    config = EvaluationConfig()
    print("\nConfiguration loaded")
    
    # 2. Generate synthetic data
    print("\nGenerating synthetic trajectories...")
    policy_trajectories = generate_synthetic_trajectories(n_trajectories=200)
    baseline_trajectories = generate_synthetic_trajectories(n_trajectories=200)
    print(f"Generated {len(policy_trajectories)} policy trajectories")
    print(f"Generated {len(baseline_trajectories)} baseline trajectories")
    
    # 3. Performance Evaluation
    print("\n" + "="*70)
    print("PERFORMANCE EVALUATION")
    print("="*70)
    perf_evaluator = PerformanceEvaluator(config)
    perf_result = perf_evaluator.evaluate(policy_trajectories)
    
    # 4. Safety Evaluation
    print("\n" + "="*70)
    print("SAFETY EVALUATION")
    print("="*70)
    safety_evaluator = SafetyEvaluator(config)
    safety_result = safety_evaluator.evaluate(policy_trajectories)
    
    # 5. Clinical Evaluation
    print("\n" + "="*70)
    print("CLINICAL EVALUATION")
    print("="*70)
    clinical_evaluator = ClinicalEvaluator(config)
    clinical_result = clinical_evaluator.evaluate(
        policy_trajectories,
        baseline_trajectories
    )
    
    # 6. Off-Policy Evaluation
    print("\n" + "="*70)
    print("OFF-POLICY EVALUATION")
    print("="*70)
    
    # Create dummy policy and behavior policy
    class DummyPolicy:
        def __call__(self, state, deterministic=False):
            return np.random.uniform(0, 1, size=2)
        
        def get_action_probability(self, state, action):
            return 0.25  # Uniform over 4 actions
    
    policy = DummyPolicy()
    behavior_policy = DummyPolicy()
    
    # Convert trajectories to Trajectory objects
    traj_objects = []
    for traj in policy_trajectories[:50]:  # Use subset for OPE
        states = np.array([list(s.values()) for s in traj['states']])
        actions = np.array([list(a.values()) for a in traj['actions']])
        rewards = np.array(traj['rewards'])
        next_states = np.array([list(s.values()) for s in traj['next_states']])
        dones = np.array(traj['dones'], dtype=float)
        
        traj_obj = Trajectory(states, actions, rewards, next_states, dones)
        traj_objects.append(traj_obj)
    
    ope_evaluator = OffPolicyEvaluator(gamma=0.99)
    ope_results = ope_evaluator.evaluate(
        policy, 
        behavior_policy,
        traj_objects,
        methods=['is', 'wis']
    )
    
    ope_evaluator.compare_methods(ope_results)
    
    # 7. Visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = EvaluationVisualizer(config)
    
    # Create comparison data
    import pandas as pd
    comparison_data = {
        'CQL': {
            'avg_return': perf_result.average_return,
            'safety_index': safety_result.safety_index,
            'adverse_events': clinical_result.adverse_event_rate
        },
        'Baseline': {
            'avg_return': perf_result.average_return * 0.8,
            'safety_index': safety_result.safety_index * 0.9,
            'adverse_events': clinical_result.adverse_event_rate * 1.2
        }
    }
    comparison_df = pd.DataFrame(comparison_data).T
    
    print("\nCreating comparison plot...")
    visualizer.plot_comparison(
        comparison_df,
        metrics=['avg_return', 'safety_index'],
        save_path='comparison.png'
    )
    
    print("\nCreating safety violations plot...")
    visualizer.plot_safety_violations(
        policy_trajectories[:50],
        variables=['glucose', 'bp_systolic'],
        save_path='safety_violations.png'
    )
    
    print("\nCreating health metrics plot...")
    visualizer.plot_health_metrics(
        policy_trajectories[:50],
        metrics=['glucose', 'bp_systolic'],
        save_path='health_metrics.png'
    )
    
    # 8. Summary Report
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nPerformance:")
    print(f"  Average Return: {perf_result.average_return:.3f}")
    print(f"  Success Rate:   {perf_result.success_rate:.3f}")
    
    print(f"\nSafety:")
    print(f"  Safety Index:         {safety_result.safety_index:.3f}")
    print(f"  Violation Rate:       {safety_result.violation_rate:.3f}")
    print(f"  Critical Violations:  {safety_result.critical_violations}")
    
    print(f"\nClinical Outcomes:")
    print(f"  Adverse Event Rate:    {clinical_result.adverse_event_rate:.3f}")
    print(f"  Goal Achievement Rate: {clinical_result.goal_achievement_rate:.3f}")
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
