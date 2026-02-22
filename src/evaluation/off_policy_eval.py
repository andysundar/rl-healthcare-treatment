"""
Off-Policy Evaluation (OPE) Methods

Implements: IS, WIS, DR, DM
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    
    def __len__(self) -> int:
        return len(self.rewards)

@dataclass
class OPEResult:
    value_estimate: float
    std_error: float
    confidence_interval: Tuple[float, float]
    method: str
    n_trajectories: int
    metadata: Dict = None

class ImportanceSampling:
    """Standard Importance Sampling estimator."""
    
    def __init__(self, gamma: float = 0.99, clip_ratio: float = 10.0):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
    def evaluate(self, policy, behavior_policy, trajectories, deterministic=False):
        values = []
        for traj in trajectories:
            importance_ratio = 1.0
            discounted_return = 0.0
            
            for t in range(len(traj)):
                pi_prob = self._get_prob(policy, traj.states[t], traj.actions[t], deterministic)
                b_prob = self._get_prob(behavior_policy, traj.states[t], traj.actions[t], False)
                ratio = np.clip(pi_prob / (b_prob + 1e-8), 0, self.clip_ratio)
                importance_ratio *= ratio
                discounted_return += (self.gamma ** t) * traj.rewards[t]
            
            values.append(importance_ratio * discounted_return)
        
        return OPEResult(
            value_estimate=np.mean(values),
            std_error=np.std(values) / np.sqrt(len(values)),
            confidence_interval=(np.mean(values) - 1.96*np.std(values)/np.sqrt(len(values)),
                                np.mean(values) + 1.96*np.std(values)/np.sqrt(len(values))),
            method='importance_sampling',
            n_trajectories=len(trajectories)
        )
    
    def _get_prob(self, policy, state, action, deterministic):
        if deterministic:
            policy_action = policy(state, deterministic=True)
            return 1.0 if np.allclose(policy_action, action) else 1e-8
        return policy.get_action_probability(state, action)

class WeightedImportanceSampling:
    """WIS with lower variance."""
    
    def __init__(self, gamma: float = 0.99, clip_ratio: float = 10.0):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
    def evaluate(self, policy, behavior_policy, trajectories, deterministic=False):
        weighted_returns = []
        weights = []
        
        for traj in trajectories:
            weight = 1.0
            traj_return = 0.0
            
            for t in range(len(traj)):
                pi_prob = self._get_prob(policy, traj.states[t], traj.actions[t], deterministic)
                b_prob = self._get_prob(behavior_policy, traj.states[t], traj.actions[t], False)
                ratio = np.clip(pi_prob / (b_prob + 1e-8), 0, self.clip_ratio)
                weight *= ratio
                traj_return += (self.gamma ** t) * traj.rewards[t]
            
            weights.append(weight)
            weighted_returns.append(weight * traj_return)
        
        total_weight = sum(weights)
        wis_estimate = sum(weighted_returns) / total_weight if total_weight > 0 else 0.0
        std_error = self._bootstrap_std(weighted_returns, weights)
        
        return OPEResult(
            value_estimate=wis_estimate,
            std_error=std_error,
            confidence_interval=(wis_estimate - 1.96*std_error, wis_estimate + 1.96*std_error),
            method='weighted_importance_sampling',
            n_trajectories=len(trajectories),
            metadata={'total_weight': total_weight}
        )
    
    def _get_prob(self, policy, state, action, deterministic):
        if deterministic:
            policy_action = policy(state, deterministic=True)
            return 1.0 if np.allclose(policy_action, action) else 1e-8
        return policy.get_action_probability(state, action)
    
    def _bootstrap_std(self, weighted_returns, weights, n_bootstrap=1000):
        n = len(weighted_returns)
        estimates = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_weighted_returns = [weighted_returns[i] for i in indices]
            boot_weights = [weights[i] for i in indices]
            total_weight = sum(boot_weights)
            if total_weight > 0:
                estimates.append(sum(boot_weighted_returns) / total_weight)
        return np.std(estimates)

class DoublyRobust:
    """DR estimator combining IS and Q-function."""
    
    def __init__(self, q_function, gamma=0.99, clip_ratio=10.0):
        self.q_function = q_function
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
    def evaluate(self, policy, behavior_policy, trajectories, action_space=None, deterministic=False):
        values = []
        
        for traj in trajectories:
            dr_value = 0.0
            for t in range(len(traj)):
                state = traj.states[t]
                action = traj.actions[t]
                reward = traj.rewards[t]
                
                pi_prob = self._get_prob(policy, state, action, deterministic)
                b_prob = self._get_prob(behavior_policy, state, action, False)
                rho = np.clip(pi_prob / (b_prob + 1e-8), 0, self.clip_ratio)
                
                q_estimate = self.q_function(state, action)
                dr_term = rho * (reward - q_estimate)
                
                if action_space is not None:
                    v_estimate = sum(self._get_prob(policy, state, a, deterministic) * 
                                   self.q_function(state, a) for a in action_space)
                else:
                    policy_action = policy(state, deterministic=True)
                    v_estimate = self.q_function(state, policy_action)
                
                dr_value += (self.gamma ** t) * (dr_term + v_estimate)
            
            values.append(dr_value)
        
        return OPEResult(
            value_estimate=np.mean(values),
            std_error=np.std(values) / np.sqrt(len(values)),
            confidence_interval=(np.mean(values) - 1.96*np.std(values)/np.sqrt(len(values)),
                                np.mean(values) + 1.96*np.std(values)/np.sqrt(len(values))),
            method='doubly_robust',
            n_trajectories=len(trajectories)
        )
    
    def _get_prob(self, policy, state, action, deterministic):
        if deterministic:
            policy_action = policy(state, deterministic=True)
            return 1.0 if np.allclose(policy_action, action) else 1e-8
        return policy.get_action_probability(state, action)

class DirectMethod:
    """Model-based OPE using Q-function."""
    
    def __init__(self, q_function, gamma=0.99):
        self.q_function = q_function
        self.gamma = gamma
        
    def evaluate(self, policy, trajectories, deterministic=True):
        values = []
        
        for traj in trajectories:
            value = 0.0
            for t in range(len(traj)):
                action = policy(traj.states[t], deterministic=deterministic)
                q_value = self.q_function(traj.states[t], action)
                value += (self.gamma ** t) * q_value
            values.append(value)
        
        return OPEResult(
            value_estimate=np.mean(values),
            std_error=np.std(values) / np.sqrt(len(values)),
            confidence_interval=(np.mean(values) - 1.96*np.std(values)/np.sqrt(len(values)),
                                np.mean(values) + 1.96*np.std(values)/np.sqrt(len(values))),
            method='direct_method',
            n_trajectories=len(trajectories)
        )

class OffPolicyEvaluator:
    """Unified OPE interface."""
    
    def __init__(self, gamma=0.99, clip_ratio=10.0, q_function=None):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.q_function = q_function
        
        self.is_evaluator = ImportanceSampling(gamma, clip_ratio)
        self.wis_evaluator = WeightedImportanceSampling(gamma, clip_ratio)
        
        if q_function is not None:
            self.dr_evaluator = DoublyRobust(q_function, gamma, clip_ratio)
            self.dm_evaluator = DirectMethod(q_function, gamma)
    
    def evaluate(self, policy, behavior_policy, trajectories, methods=['wis'], **kwargs):
        results = {}
        
        for method in methods:
            if method == 'is':
                results['is'] = self.is_evaluator.evaluate(policy, behavior_policy, trajectories, **kwargs)
            elif method == 'wis':
                results['wis'] = self.wis_evaluator.evaluate(policy, behavior_policy, trajectories, **kwargs)
            elif method == 'dr' and self.q_function:
                results['dr'] = self.dr_evaluator.evaluate(policy, behavior_policy, trajectories, **kwargs)
            elif method == 'dm' and self.q_function:
                results['dm'] = self.dm_evaluator.evaluate(policy, trajectories, **kwargs)
        
        return results
    
    def compare_methods(self, results):
        print("\n" + "="*70)
        print("OFF-POLICY EVALUATION COMPARISON")
        print("="*70)
        print(f"{'Method':<10} {'Value':<12} {'Std Error':<12} {'95% CI':<25}")
        print("-"*70)
        
        for method_name, result in results.items():
            ci_str = f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
            print(f"{method_name.upper():<10} {result.value_estimate:<12.3f} "
                  f"{result.std_error:<12.3f} {ci_str:<25}")
        
        print("="*70 + "\n")
