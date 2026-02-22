"""
Off-Policy Evaluation Methods Comparison

Demonstrates all four OPE methods on synthetic data.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from evaluation import OffPolicyEvaluator, Trajectory


class SimplePolicy:
    """Simple Gaussian policy for demonstration."""
    
    def __init__(self, mean=0.5, std=0.2):
        self.mean = mean
        self.std = std
    
    def __call__(self, state, deterministic=False):
        if deterministic:
            return np.array([self.mean, self.mean])
        return np.random.normal(self.mean, self.std, size=2)
    
    def get_action_probability(self, state, action):
        # Assuming independent Gaussian
        prob = 1.0
        for a in action:
            prob *= np.exp(-0.5 * ((a - self.mean) / self.std) ** 2)
            prob /= (self.std * np.sqrt(2 * np.pi))
        return prob


class SimpleQFunction:
    """Simple Q-function for demonstration."""
    
    def __init__(self):
        pass
    
    def __call__(self, state, action):
        # Dummy Q-function: higher for actions close to 0.5
        return -np.sum((action - 0.5) ** 2) + np.random.randn() * 0.1


def generate_trajectories(n_traj=50, episode_length=30):
    """Generate synthetic trajectories."""
    trajectories = []
    
    for _ in range(n_traj):
        states = np.random.randn(episode_length, 4)
        actions = np.random.uniform(0, 1, size=(episode_length, 2))
        rewards = np.random.randn(episode_length)
        next_states = np.random.randn(episode_length, 4)
        dones = np.zeros(episode_length)
        dones[-1] = 1.0
        
        traj = Trajectory(states, actions, rewards, next_states, dones)
        trajectories.append(traj)
    
    return trajectories


def main():
    print("="*70)
    print("OPE METHODS COMPARISON")
    print("="*70)
    
    # Generate data
    print("\nGenerating synthetic trajectories...")
    trajectories = generate_trajectories(n_traj=100)
    print(f"Generated {len(trajectories)} trajectories")
    
    # Create policies
    print("\nCreating policies...")
    target_policy = SimplePolicy(mean=0.5, std=0.2)
    behavior_policy = SimplePolicy(mean=0.4, std=0.3)
    q_function = SimpleQFunction()
    
    # Initialize evaluator
    ope_evaluator = OffPolicyEvaluator(
        gamma=0.99,
        clip_ratio=10.0,
        q_function=q_function
    )
    
    # Evaluate with all methods
    print("\nRunning OPE with all methods...")
    print("-"*70)
    
    methods = ['is', 'wis', 'dr', 'dm']
    results = ope_evaluator.evaluate(
        target_policy,
        behavior_policy,
        trajectories,
        methods=methods
    )
    
    # Compare methods
    ope_evaluator.compare_methods(results)
    
    # Detailed analysis
    print("\nDETAILED ANALYSIS")
    print("="*70)
    
    for method_name, result in results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Value Estimate:   {result.value_estimate:.4f}")
        print(f"  Standard Error:   {result.std_error:.4f}")
        print(f"  95% CI:           [{result.confidence_interval[0]:.4f}, "
              f"{result.confidence_interval[1]:.4f}]")
        print(f"  CI Width:         {result.confidence_interval[1] - result.confidence_interval[0]:.4f}")
        
        if result.metadata:
            print(f"  Metadata:         {result.metadata}")
    
    # Compare variance
    print("\n" + "="*70)
    print("VARIANCE COMPARISON")
    print("="*70)
    
    variances = {name: res.std_error**2 for name, res in results.items()}
    sorted_methods = sorted(variances.items(), key=lambda x: x[1])
    
    print("\nMethods by variance (lower is better):")
    for rank, (method, var) in enumerate(sorted_methods, 1):
        print(f"  {rank}. {method.upper():<10}: {var:.6f}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
For Healthcare RL Offline Evaluation:

1. WEIGHTED IMPORTANCE SAMPLING (WIS)
   - Lower variance than standard IS
   - No model needed
   - Good for first-pass evaluation
   - Use when: Limited data, no Q-function

2. DOUBLY ROBUST (DR)
   - Best of both worlds
   - Robust to model errors
   - Requires Q-function
   - Use when: Have Q-network, need accuracy

3. DIRECT METHOD (DM)
   - Lowest variance
   - Biased if Q-function wrong
   - Fast computation
   - Use when: Strong confidence in Q-function

4. IMPORTANCE SAMPLING (IS)
   - Unbiased
   - High variance
   - Simple
   - Use for: Sanity checks, small datasets

RECOMMENDED FOR YOUR THESIS:
- Primary: WIS (robust, practical)
- Secondary: DR (if Q-function available)
- Report both for completeness
    """)
    
    print("="*70)
    print("Comparison complete!")


if __name__ == "__main__":
    main()
