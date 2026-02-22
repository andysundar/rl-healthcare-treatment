"""
Environment Testing Utilities
==============================
Tools for testing and validating healthcare RL environments.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from gymnasium import Env
import time


def test_environment(env: Env, 
                     n_episodes: int = 10,
                     max_steps_per_episode: Optional[int] = None,
                     render: bool = False,
                     verbose: bool = True) -> Dict[str, Any]:
    """
    Test basic environment functionality.
    
    Args:
        env: Gymnasium environment to test
        n_episodes: Number of test episodes
        max_steps_per_episode: Maximum steps per episode (uses env default if None)
        render: Whether to render episodes
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary of test results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing Environment: {env.__class__.__name__}")
        print(f"{'='*70}\n")
    
    episode_rewards = []
    episode_lengths = []
    safety_violations = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Validate initial state
        assert env.observation_space.contains(state), \
            f"Initial state {state} not in observation space"
        
        while not done:
            # Sample random action
            action = env.action_space.sample()
            
            # Validate action
            assert env.action_space.contains(action), \
                f"Action {action} not in action space"
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Validate outputs
            assert env.observation_space.contains(next_state), \
                f"Next state {next_state} not in observation space"
            assert isinstance(reward, (int, float)), \
                f"Reward {reward} is not a number"
            assert isinstance(terminated, bool), \
                f"Terminated {terminated} is not boolean"
            assert isinstance(truncated, bool), \
                f"Truncated {truncated} is not boolean"
            assert isinstance(info, dict), \
                f"Info {info} is not a dictionary"
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
                time.sleep(0.05)  # Small delay for readability
            
            # Check max steps
            if max_steps_per_episode and steps >= max_steps_per_episode:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Get episode metrics if available
        if hasattr(env, 'get_episode_metrics'):
            metrics = env.get_episode_metrics()
            safety_violations.append(metrics.get('safety_violations', 0))
        
        if verbose:
            print(f"Episode {episode + 1:3d}: "
                  f"Steps={steps:4d}, "
                  f"Reward={total_reward:8.2f}, "
                  f"Safety Violations={safety_violations[-1] if safety_violations else 0}")
    
    # Compute statistics
    results = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    if safety_violations:
        results['mean_safety_violations'] = np.mean(safety_violations)
        results['total_safety_violations'] = sum(safety_violations)
    
    if verbose:
        print(f"\n{'-'*70}")
        print(f"Test Results:")
        print(f"{'-'*70}")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
        if safety_violations:
            print(f"Total Safety Violations: {results['total_safety_violations']}")
        print(f"{'='*70}\n")
    
    return results


def test_determinism(env: Env, 
                     seed: int = 42, 
                     n_steps: int = 100,
                     verbose: bool = True) -> bool:
    """
    Test if environment is deterministic with same seed.
    
    Args:
        env: Environment to test
        seed: Random seed
        n_steps: Number of steps to test
        verbose: Whether to print results
        
    Returns:
        is_deterministic: Whether environment is deterministic
    """
    if verbose:
        print(f"\nTesting determinism with seed={seed}...")
    
    # Run 1
    states1 = []
    rewards1 = []
    
    state, _ = env.reset(seed=seed)
    states1.append(state.copy())
    
    for _ in range(n_steps):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        states1.append(state.copy())
        rewards1.append(reward)
        
        if terminated or truncated:
            break
    
    # Run 2 (with same seed)
    states2 = []
    rewards2 = []
    
    state, _ = env.reset(seed=seed)
    states2.append(state.copy())
    
    for _ in range(len(states1) - 1):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        states2.append(state.copy())
        rewards2.append(reward)
        
        if terminated or truncated:
            break
    
    # Compare
    states_match = all(np.allclose(s1, s2) for s1, s2 in zip(states1, states2))
    rewards_match = all(np.isclose(r1, r2) for r1, r2 in zip(rewards1, rewards2))
    
    is_deterministic = states_match and rewards_match
    
    if verbose:
        if is_deterministic:
            print("✓ Environment is deterministic")
        else:
            print("✗ Environment is NOT deterministic")
    
    return is_deterministic


def test_action_effects(env: Env,
                       n_samples: int = 100,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Test if different actions lead to different outcomes.
    
    Args:
        env: Environment to test
        n_samples: Number of state samples to test
        verbose: Whether to print results
        
    Returns:
        results: Dictionary of test results
    """
    if verbose:
        print(f"\nTesting action effects ({n_samples} samples)...")
    
    different_outcomes = 0
    
    for _ in range(n_samples):
        # Reset environment
        state, _ = env.reset()
        
        # Try two different actions
        action1 = env.action_space.sample()
        
        # Get second action that's different from first
        action2 = env.action_space.sample()
        max_attempts = 100
        attempts = 0
        while np.allclose(action1, action2) and attempts < max_attempts:
            action2 = env.action_space.sample()
            attempts += 1
        
        # Reset to same state
        state1, _ = env.reset()
        
        # Take actions
        next_state1, reward1, _, _, _ = env.step(action1)
        
        state2, _ = env.reset()
        next_state2, reward2, _, _, _ = env.step(action2)
        
        # Check if outcomes differ
        if not np.allclose(next_state1, next_state2) or not np.isclose(reward1, reward2):
            different_outcomes += 1
    
    action_effect_rate = different_outcomes / n_samples
    
    if verbose:
        print(f"Action effect rate: {action_effect_rate:.1%}")
        if action_effect_rate > 0.9:
            print("✓ Actions have strong effects on outcomes")
        elif action_effect_rate > 0.5:
            print("⚠ Actions have moderate effects on outcomes")
        else:
            print("✗ Actions have weak effects on outcomes")
    
    return {
        'action_effect_rate': action_effect_rate,
        'n_samples': n_samples,
        'different_outcomes': different_outcomes
    }


def test_safety_constraints(env: Env,
                           n_episodes: int = 20,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Test safety constraint detection in environment.
    
    Args:
        env: Environment to test
        n_episodes: Number of episodes
        verbose: Whether to print results
        
    Returns:
        results: Safety test results
    """
    if not hasattr(env, '_is_unsafe_state'):
        if verbose:
            print("\n⚠ Environment does not implement safety constraints")
        return {'has_safety': False}
    
    if verbose:
        print(f"\nTesting safety constraints ({n_episodes} episodes)...")
    
    total_steps = 0
    unsafe_steps = 0
    episodes_with_violations = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_violations = 0
        
        while not done:
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            
            if env._is_unsafe_state(next_state):
                unsafe_steps += 1
                episode_violations += 1
            
            total_steps += 1
            done = terminated or truncated
        
        if episode_violations > 0:
            episodes_with_violations += 1
    
    results = {
        'has_safety': True,
        'total_steps': total_steps,
        'unsafe_steps': unsafe_steps,
        'safety_violation_rate': unsafe_steps / total_steps if total_steps > 0 else 0,
        'episodes_with_violations': episodes_with_violations,
        'episode_violation_rate': episodes_with_violations / n_episodes
    }
    
    if verbose:
        print(f"Safety violation rate: {results['safety_violation_rate']:.1%}")
        print(f"Episodes with violations: {episodes_with_violations}/{n_episodes}")
    
    return results


def benchmark_environment(env: Env,
                         n_episodes: int = 100,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Benchmark environment performance (speed).
    
    Args:
        env: Environment to benchmark
        n_episodes: Number of episodes
        verbose: Whether to print results
        
    Returns:
        results: Benchmark results
    """
    if verbose:
        print(f"\nBenchmarking environment ({n_episodes} episodes)...")
    
    total_steps = 0
    start_time = time.time()
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            total_steps += 1
            done = terminated or truncated
    
    elapsed_time = time.time() - start_time
    steps_per_second = total_steps / elapsed_time
    
    results = {
        'total_steps': total_steps,
        'elapsed_time': elapsed_time,
        'steps_per_second': steps_per_second,
        'episodes_per_second': n_episodes / elapsed_time
    }
    
    if verbose:
        print(f"Steps per second: {steps_per_second:.0f}")
        print(f"Episodes per second: {results['episodes_per_second']:.1f}")
    
    return results


def comprehensive_test_suite(env: Env,
                             n_episodes: int = 10,
                             verbose: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive test suite on environment.
    
    Args:
        env: Environment to test
        n_episodes: Number of episodes for relevant tests
        verbose: Whether to print results
        
    Returns:
        results: Comprehensive test results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE TEST SUITE: {env.__class__.__name__}")
        print(f"{'='*70}")
    
    results = {}
    
    # Basic functionality
    results['functionality'] = test_environment(env, n_episodes, verbose=verbose)
    
    # Determinism
    results['determinism'] = {
        'is_deterministic': test_determinism(env, verbose=verbose)
    }
    
    # Action effects
    results['action_effects'] = test_action_effects(env, n_samples=50, verbose=verbose)
    
    # Safety
    results['safety'] = test_safety_constraints(env, n_episodes=n_episodes, verbose=verbose)
    
    # Performance
    results['benchmark'] = benchmark_environment(env, n_episodes=n_episodes, verbose=verbose)
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        print(f"✓ Basic functionality: PASSED")
        print(f"{'✓' if results['determinism']['is_deterministic'] else '✗'} Determinism: "
              f"{'PASSED' if results['determinism']['is_deterministic'] else 'FAILED'}")
        print(f"{'✓' if results['action_effects']['action_effect_rate'] > 0.5 else '✗'} "
              f"Action effects: {'STRONG' if results['action_effects']['action_effect_rate'] > 0.9 else 'MODERATE' if results['action_effects']['action_effect_rate'] > 0.5 else 'WEAK'}")
        print(f"✓ Safety constraints: {'IMPLEMENTED' if results['safety']['has_safety'] else 'NOT IMPLEMENTED'}")
        print(f"✓ Performance: {results['benchmark']['steps_per_second']:.0f} steps/sec")
        print(f"{'='*70}\n")
    
    return results


class PolicyEvaluator:
    """Evaluate a policy in an environment."""
    
    def __init__(self, env: Env):
        """
        Initialize policy evaluator.
        
        Args:
            env: Environment to evaluate in
        """
        self.env = env
    
    def evaluate(self,
                policy: Callable,
                n_episodes: int = 100,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate a policy.
        
        Args:
            policy: Policy function that takes state and returns action
            n_episodes: Number of evaluation episodes
            verbose: Whether to print progress
            
        Returns:
            results: Evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        safety_violations = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            violations = 0
            
            while not done:
                action = policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Check safety
                if hasattr(self.env, '_is_unsafe_state'):
                    if self.env._is_unsafe_state(next_state):
                        violations += 1
                
                total_reward += reward
                steps += 1
                done = terminated or truncated
                state = next_state
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            safety_violations.append(violations)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Mean Reward={np.mean(episode_rewards):.2f}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'total_safety_violations': sum(safety_violations),
            'safety_violation_rate': sum(safety_violations) / sum(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'safety_violations': safety_violations
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("POLICY EVALUATION RESULTS")
            print(f"{'='*70}")
            print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Mean Episode Length: {results['mean_episode_length']:.1f}")
            print(f"Safety Violations: {results['total_safety_violations']} ({results['safety_violation_rate']:.1%})")
            print(f"{'='*70}\n")
        
        return results
