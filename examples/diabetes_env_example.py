"""
Example: Diabetes Management Environment
=========================================
Demonstrates usage of the diabetes management environment with Bergman model.
"""

import numpy as np
import sys
sys.path.append('..')

from environments import (
    DiabetesManagementEnv,
    DiabetesEnvConfig,
    test_environment,
    comprehensive_test_suite
)


def random_policy_example():
    """Run episodes with random policy."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Random Policy")
    print("="*70)
    
    # Create environment
    config = DiabetesEnvConfig(
        max_steps=30,  # 30 days
        dt=1440.0,     # 1 day timesteps
    )
    env = DiabetesManagementEnv(config=config, render_mode='human')
    
    # Run one episode
    state, info = env.reset(seed=42)
    print(f"\nInitial state: {state}")
    print(f"Initial info: {info}\n")
    
    total_reward = 0
    for step in range(5):  # Show first 5 steps
        action = env.action_space.sample()
        print(f"\nAction: insulin={action[0]:.1f} units, "
              f"meal_adj={action[1]:.2f}, reminders={int(action[2])}")
        
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        print(f"Reward: {reward:.2f}")
        
        if terminated or truncated:
            print("\nEpisode terminated!")
            break
    
    print(f"\nTotal reward (first 5 steps): {total_reward:.2f}")
    env.close()


def simple_rule_based_policy():
    """Demonstrate a simple rule-based policy."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Rule-Based Policy")
    print("="*70)
    
    def rule_based_insulin(glucose, target_range=(80, 130)):
        """
        Simple rule: adjust insulin based on glucose level.
        """
        target_mid = (target_range[0] + target_range[1]) / 2
        
        if glucose < target_range[0]:
            # Hypoglycemic - reduce insulin
            return max(0, 5 * (glucose / target_mid))
        elif glucose > target_range[1]:
            # Hyperglycemic - increase insulin
            return min(50, 20 * (glucose / target_mid - 1))
        else:
            # In range - maintain baseline
            return 10
    
    env = DiabetesManagementEnv()
    
    # Run episode
    state, _ = env.reset(seed=42)
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 30:  # 30 days
        glucose = state[0]
        
        # Rule-based action
        insulin_dose = rule_based_insulin(glucose)
        meal_adjustment = 0.0  # No meal adjustment
        reminders = 2  # Moderate reminder frequency
        
        action = np.array([insulin_dose, meal_adjustment, reminders], dtype=np.float32)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = next_state
        step += 1
        
        if step % 7 == 0:  # Print weekly summary
            print(f"\nDay {step}: Glucose={glucose:.1f} mg/dL, "
                  f"Insulin={insulin_dose:.1f} units, Reward={reward:.2f}")
    
    print(f"\n30-day total reward: {total_reward:.2f}")
    
    # Get physiological state
    final_state = env.get_physiological_state()
    print("\nFinal physiological state:")
    for key, value in final_state.items():
        print(f"  {key}: {value}")
    
    env.close()


def test_patient_heterogeneity():
    """Test with different patient parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Patient Heterogeneity")
    print("="*70)
    
    # Create environments for different patients
    patient_ids = [0, 1, 2]
    
    for pid in patient_ids:
        print(f"\n--- Patient {pid} ---")
        env = DiabetesManagementEnv(patient_id=pid)
        
        # Test with same random policy
        state, _ = env.reset(seed=42)
        total_reward = 0
        
        for _ in range(30):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Total reward: {total_reward:.2f}")
        
        # Get safety metrics
        metrics = env.get_episode_metrics()
        print(f"Safety violations: {metrics.get('safety_violations', 0)}")
        print(f"Episode length: {metrics.get('episode_length', 0)}")
        
        env.close()


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comprehensive Testing")
    print("="*70)
    
    env = DiabetesManagementEnv()
    
    # Run comprehensive test suite
    results = comprehensive_test_suite(env, n_episodes=10, verbose=True)
    
    env.close()
    
    return results


def demonstrate_safety_constraints():
    """Demonstrate safety constraint detection."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Safety Constraints")
    print("="*70)
    
    env = DiabetesManagementEnv()
    
    # Try to induce unsafe states with extreme actions
    state, _ = env.reset(seed=42)
    
    print("\nAttempting unsafe action (excessive insulin)...")
    unsafe_action = np.array([100.0, 0.0, 0.0], dtype=np.float32)  # Max insulin
    
    for step in range(10):
        next_state, reward, terminated, truncated, info = env.step(unsafe_action)
        
        glucose = next_state[0]
        is_unsafe = env._is_unsafe_state(next_state)
        
        print(f"Step {step+1}: Glucose={glucose:.1f} mg/dL, "
              f"Unsafe={'YES ⚠️' if is_unsafe else 'NO'}, "
              f"Reward={reward:.2f}")
        
        if terminated:
            print("\n⚠️ Episode terminated due to critical glucose level!")
            break
        
        if truncated:
            break
    
    # Get safety metrics
    metrics = env.get_episode_metrics()
    print(f"\nSafety violations: {metrics.get('safety_violations', 0)}")
    print(f"Safety violation rate: {metrics.get('safety_violation_rate', 0):.1%}")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DIABETES MANAGEMENT ENVIRONMENT EXAMPLES")
    print("="*70)
    
    # Run all examples
    random_policy_example()
    simple_rule_based_policy()
    test_patient_heterogeneity()
    demonstrate_safety_constraints()
    run_comprehensive_tests()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")
