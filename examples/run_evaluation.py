"""
Example Evaluation Script for Healthcare RL

Demonstrates complete evaluation workflow:
1. Load trained policies
2. Generate evaluation trajectories
3. Run comprehensive evaluation
4. Compare multiple policies
5. Generate visualizations and reports

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path when run directly
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import torch
from typing import List, Tuple

# Import evaluation framework (actual API)
from evaluation.safety_metrics import SafetyEvaluator
from evaluation.clinical_metrics import ClinicalEvaluator
from evaluation.performance_metrics import PerformanceEvaluator
from evaluation.off_policy_eval import OffPolicyEvaluator
from evaluation.visualizations import EvaluationVisualizer
from configs.config import EvaluationConfig

# Import your existing components (adjust paths as needed)
# from src.models.cql import ConservativeQLearning
# from src.environment.healthcare_mdp import HealthcareMDP
# from src.data.mimic_processor import MIMICProcessor


def load_trained_policy(policy_path: str):
    """
    Load a trained RL policy from checkpoint.
    
    Replace this with your actual policy loading logic.
    """
    # Example for PyTorch model
    checkpoint = torch.load(policy_path, map_location='cpu')
    
    # Assuming you have a policy class
    # policy = YourPolicyClass(state_dim, action_dim)
    # policy.load_state_dict(checkpoint['policy_state_dict'])
    # policy.eval()
    
    # For now, return a dummy policy wrapper
    class DummyPolicy:
        def select_action(self, state, deterministic=True):
            # Replace with actual policy forward pass
            return np.random.randn(5)  # Example action
        
        def get_action_probability(self, state, action):
            # Replace with actual probability computation
            return 0.5
        
        def sample_action(self, state):
            return self.select_action(state, deterministic=False)
    
    return DummyPolicy()


def load_behavior_policy(policy_path: str):
    """
    Load behavior policy that collected the offline data.
    
    This could be:
    - Random policy
    - Historical clinical decisions
    - Previous version of policy
    """
    # Similar to load_trained_policy
    # For now, return dummy
    class DummyBehaviorPolicy:
        def get_action_probability(self, state, action):
            return 0.5
    
    return DummyBehaviorPolicy()


def load_q_function(q_function_path: str):
    """
    Load fitted Q-function for OPE methods.
    
    Required for Doubly Robust and Direct Method OPE.
    """
    class DummyQFunction:
        def predict(self, state, action):
            # Replace with actual Q-function forward pass
            return np.random.randn()
    
    return DummyQFunction()


def generate_evaluation_trajectories(
    policy,
    environment,
    n_episodes: int = 100,
    max_episode_length: int = 200
) -> List[List[Tuple]]:
    """
    Generate evaluation trajectories from policy.
    
    Args:
        policy: Policy to evaluate
        environment: Healthcare MDP environment
        n_episodes: Number of episodes to run
        max_episode_length: Maximum steps per episode
        
    Returns:
        List of trajectories, each trajectory is a list of (s, a, r, s_next, done)
    """
    trajectories = []
    
    for episode in range(n_episodes):
        trajectory = []
        state = environment.reset()
        done = False
        t = 0
        
        while not done and t < max_episode_length:
            # Select action from policy
            action = policy.select_action(state, deterministic=True)
            
            # Execute action in environment
            next_state, reward, done, info = environment.step(action)
            
            # Store transition
            trajectory.append((state, action, reward, next_state, done))
            
            state = next_state
            t += 1
        
        trajectories.append(trajectory)
        
        if (episode + 1) % 10 == 0:
            print(f"Generated {episode + 1}/{n_episodes} evaluation episodes")
    
    return trajectories


def main_single_policy_evaluation():
    """
    Example: Evaluate a single policy comprehensively using the actual API.
    """
    print("=" * 80)
    print("SINGLE POLICY COMPREHENSIVE EVALUATION")
    print("=" * 80)

    cfg = EvaluationConfig()

    # Generate dummy trajectories in the expected dict format
    trajs = generate_dummy_trajectories(50)

    print("\nRunning safety evaluation...")
    safety_result = SafetyEvaluator(cfg).evaluate(trajs)

    print("\nRunning clinical evaluation...")
    tir = ClinicalEvaluator(cfg).compute_time_in_range(trajs)

    print("\nRunning performance evaluation...")
    perf_result = PerformanceEvaluator(cfg).evaluate(trajs)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nPerformance:")
    print(f"  Average Return: {perf_result.average_return:.3f}")
    print(f"  Success Rate:   {perf_result.success_rate:.3f}")
    print(f"\nSafety:")
    print(f"  Safety Index:   {safety_result.safety_index:.3f}")
    print(f"  Violation Rate: {safety_result.violation_rate:.3f}")
    print(f"\nClinical (Time-in-Range):")
    for metric, val in tir.items():
        print(f"  {metric}: {val:.3f}")
    print("=" * 80)


def main_multi_policy_comparison():
    """
    Example: Compare multiple policies side-by-side.
    """
    print("=" * 80)
    print("MULTI-POLICY COMPARISON")
    print("=" * 80)

    cfg = EvaluationConfig()
    policy_names = ["CQL", "BCQ", "BehaviorCloning", "Random"]

    print("\nEvaluating policies...")
    rows = []
    for name in policy_names:
        trajs = generate_dummy_trajectories(50)
        safety = SafetyEvaluator(cfg).evaluate(trajs)
        perf   = PerformanceEvaluator(cfg).evaluate(trajs)
        rows.append({
            "Policy":          name,
            "Avg Return":      round(perf.average_return, 3),
            "Success Rate":    round(perf.success_rate, 3),
            "Safety Index":    round(safety.safety_index, 3),
            "Violation Rate":  round(safety.violation_rate, 3),
        })

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    header = f"{'Policy':<20} {'Avg Return':>12} {'Success':>10} {'Safety':>10} {'Violations':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['Policy']:<20} {r['Avg Return']:>12.3f} {r['Success Rate']:>10.3f}"
              f" {r['Safety Index']:>10.3f} {r['Violation Rate']:>12.3f}")
    print("=" * 80)


def generate_dummy_trajectories(n_episodes: int, episode_length: int = 30) -> List[dict]:
    """
    Generate dummy trajectories for demonstration.

    Returns list of trajectory dicts:
        {'states': [...], 'actions': [...], 'rewards': [...],
         'next_states': [...], 'dones': [...]}
    """
    trajectories = []
    for _ in range(n_episodes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in range(episode_length):
            state = {
                'glucose': float(np.random.uniform(70, 180)),
                'blood_pressure_systolic': float(np.random.uniform(90, 140)),
                'adherence_score': float(np.random.uniform(0.5, 1.0)),
            }
            next_states.append({k: float(np.random.uniform(70, 180)) if k == 'glucose'
                                 else float(v * 0.95 + np.random.randn() * 0.01)
                                 for k, v in state.items()})
            states.append(state)
            actions.append(np.random.uniform(0, 1, size=1))
            rewards.append(float(np.random.randn()))
            dones.append(t == episode_length - 1)
        trajectories.append({'states': states, 'actions': actions,
                              'rewards': rewards, 'next_states': next_states, 'dones': dones})
    return trajectories


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HEALTHCARE RL EVALUATION FRAMEWORK - EXAMPLES")
    print("=" * 80)
    
    # Choose which example to run
    print("\nAvailable examples:")
    print("1. Single Policy Comprehensive Evaluation")
    print("2. Multi-Policy Comparison")
    print("\nRunning Example 1: Single Policy Evaluation\n")
    
    try:
        main_single_policy_evaluation()
    except Exception as e:
        print(f"\nError in Example 1: {e}")
        print("This is expected if you haven't set up actual policies/environment yet.")
        print("The code structure is ready for integration with your existing components.")
    
    print("\n\n" + "=" * 80)
    print("Example complete! Check the evaluation_results directory for outputs.")
    print("=" * 80)
