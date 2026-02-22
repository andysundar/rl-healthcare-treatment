"""
Example: Medication Adherence Environment
==========================================
Demonstrates usage of the medication adherence optimization environment.
"""

import numpy as np
import sys
sys.path.append('..')

from environments import (
    MedicationAdherenceEnv,
    AdherenceEnvConfig,
    test_environment,
    PolicyEvaluator
)


def random_policy_example():
    """Run episodes with random policy."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Random Reminder Policy")
    print("="*70)
    
    # Create environment
    config = AdherenceEnvConfig(max_steps=30)  # 30 days
    env = MedicationAdherenceEnv(config=config, render_mode='human')
    
    # Run one episode
    state, info = env.reset(seed=42)
    print(f"\nInitial state: {state}")
    print(f"Initial info: {info}\n")
    
    total_reward = 0
    for step in range(10):  # Show first 10 days
        action = env.action_space.sample()
        
        reminder_types = ['none', 'sms', 'call', 'app', 'visit']
        education_types = ['none', 'benefits', 'side_effects', 'support']
        
        print(f"\nDay {step+1} Action: {reminder_types[action[0]]}, {education_types[action[1]]}")
        
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        print(f"Reward: {reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward (first 10 days): {total_reward:.2f}")
    env.close()


def adaptive_reminder_policy():
    """Demonstrate an adaptive reminder policy."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Adaptive Reminder Policy")
    print("="*70)
    
    def adaptive_policy(state):
        """
        Adaptive policy based on current adherence level.
        - Low adherence → more intensive intervention
        - High adherence → minimal intervention
        """
        adherence = state[0]
        days_since_reminder = state[1]
        satisfaction = state[2]
        side_effects = state[3]
        
        # Reminder type decision
        if adherence < 0.5:
            # Very low adherence - personal visit or call
            reminder_type = 2 if days_since_reminder > 7 else 4  # call or visit
        elif adherence < 0.7:
            # Low adherence - call or SMS
            reminder_type = 1 if days_since_reminder > 3 else 2  # sms or call
        elif adherence < 0.85:
            # Moderate adherence - app notification
            reminder_type = 3 if days_since_reminder > 5 else 0  # app or none
        else:
            # High adherence - minimal intervention
            reminder_type = 0 if days_since_reminder < 14 else 3  # none or app
        
        # Educational content decision
        if side_effects > 0.3:
            education_type = 2  # side_effects education
        elif satisfaction < 0.6:
            education_type = 1  # benefits education
        elif adherence < 0.7:
            education_type = 3  # community support
        else:
            education_type = 0  # none
        
        return np.array([reminder_type, education_type])
    
    env = MedicationAdherenceEnv()
    
    # Evaluate adaptive policy
    evaluator = PolicyEvaluator(env)
    results = evaluator.evaluate(adaptive_policy, n_episodes=50, verbose=True)
    
    print("\nAdaptive Policy Performance:")
    print(f"Mean adherence improvement: {results['mean_reward']:.2f}")
    print(f"Mean episode length: {results['mean_episode_length']:.1f} days")
    
    env.close()


def compare_reminder_strategies():
    """Compare different reminder strategies."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Reminder Strategies")
    print("="*70)
    
    strategies = {
        'No Reminders': lambda s: np.array([0, 0]),
        'Daily SMS': lambda s: np.array([1, 0]),
        'Weekly Call': lambda s: np.array([2 if s[1] >= 7 else 0, 0]),
        'App Notifications': lambda s: np.array([3, 0]),
        'Adaptive': lambda s: np.array([
            2 if s[0] < 0.6 else (3 if s[1] > 7 else 0),
            2 if s[3] > 0.3 else 0
        ])
    }
    
    env = MedicationAdherenceEnv()
    evaluator = PolicyEvaluator(env)
    
    results_summary = []
    
    for strategy_name, policy in strategies.items():
        print(f"\nTesting: {strategy_name}")
        results = evaluator.evaluate(policy, n_episodes=30, verbose=False)
        
        results_summary.append({
            'strategy': strategy_name,
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'safety_violations': results['total_safety_violations']
        })
        
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Safety Violations: {results['total_safety_violations']}")
    
    # Print summary table
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Strategy':<25} {'Mean Reward':<15} {'Std Reward':<15} {'Safety Violations':<15}")
    print("-"*70)
    
    for res in sorted(results_summary, key=lambda x: x['mean_reward'], reverse=True):
        print(f"{res['strategy']:<25} {res['mean_reward']:<15.2f} "
              f"{res['std_reward']:<15.2f} {res['safety_violations']:<15}")
    
    env.close()


def test_patient_heterogeneity():
    """Test with different patient types."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Patient Heterogeneity")
    print("="*70)
    
    # Simple policy
    policy = lambda s: np.array([1 if s[1] > 3 else 0, 0])  # SMS every 3 days
    
    patient_results = []
    
    for patient_id in range(5):
        print(f"\n--- Patient {patient_id} ---")
        env = MedicationAdherenceEnv(patient_id=patient_id)
        
        state, _ = env.reset(seed=42)
        total_reward = 0
        
        for day in range(30):
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        metrics = env.get_adherence_metrics()
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average adherence: {metrics['average_adherence']:.2%}")
        print(f"Final adherence: {metrics['current_adherence']:.2%}")
        print(f"Adherence trend: {metrics['adherence_trend']:.4f}")
        print(f"Reminder efficiency: {metrics['reminder_efficiency']:.2f}")
        
        patient_results.append({
            'patient_id': patient_id,
            'total_reward': total_reward,
            'avg_adherence': metrics['average_adherence'],
            'final_adherence': metrics['current_adherence']
        })
        
        env.close()
    
    # Summary
    print("\n" + "-"*70)
    print("Patient Heterogeneity Summary:")
    avg_reward = np.mean([r['total_reward'] for r in patient_results])
    avg_adherence = np.mean([r['avg_adherence'] for r in patient_results])
    print(f"Average reward across patients: {avg_reward:.2f}")
    print(f"Average adherence across patients: {avg_adherence:.2%}")


def demonstrate_adherence_dynamics():
    """Demonstrate adherence dynamics over time."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Adherence Dynamics")
    print("="*70)
    
    env = MedicationAdherenceEnv()
    
    # Different intervention intensities
    interventions = {
        'Minimal': np.array([0, 0]),  # No reminders
        'Low': np.array([1, 0]),      # SMS only
        'Medium': np.array([2, 1]),   # Call + benefits education
        'High': np.array([4, 3])      # Visit + support
    }
    
    for intervention_name, action in interventions.items():
        print(f"\n--- {intervention_name} Intervention ---")
        
        state, _ = env.reset(seed=42)
        adherence_trajectory = [state[0]]
        
        for day in range(30):
            state, reward, _, _, _ = env.step(action)
            adherence_trajectory.append(state[0])
        
        # Print adherence change
        initial_adherence = adherence_trajectory[0]
        final_adherence = adherence_trajectory[-1]
        change = final_adherence - initial_adherence
        
        print(f"Initial adherence: {initial_adherence:.2%}")
        print(f"Final adherence: {final_adherence:.2%}")
        print(f"Change: {change:+.2%}")
        
        # Trend
        if change > 0.1:
            trend = "Strong Improvement ✓"
        elif change > 0:
            trend = "Slight Improvement"
        elif change > -0.1:
            trend = "Stable"
        else:
            trend = "Declining ⚠️"
        print(f"Trend: {trend}")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEDICATION ADHERENCE ENVIRONMENT EXAMPLES")
    print("="*70)
    
    # Run all examples
    random_policy_example()
    adaptive_reminder_policy()
    compare_reminder_strategies()
    test_patient_heterogeneity()
    demonstrate_adherence_dynamics()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")
