"""
Replay Buffer for Offline Reinforcement Learning

This module implements experience replay buffers for storing and sampling
transitions in offline RL. Supports standard replay and prioritized experience replay.

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Standard replay buffer for offline RL.
    
    Stores transitions (s, a, r, s', done) and provides efficient sampling
    for batch training. Implements FIFO replacement when capacity is reached.
    
    Memory layout:
        - Fixed capacity circular buffer
        - Efficient numpy array storage
        - Random sampling without replacement
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = 'cpu'
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Pointer and size
        self.ptr = 0
        self.size = 0
        
        logger.info(f"Initialized ReplayBuffer with capacity {capacity}")
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state [state_dim]
            action: Action taken [action_dim]
            reward: Reward received (scalar)
            next_state: Next state [state_dim]
            done: Whether episode terminated (bool)
        """
        # Store transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary containing batched transitions:
                - states: [batch_size, state_dim]
                - actions: [batch_size, action_dim]
                - rewards: [batch_size, 1]
                - next_states: [batch_size, state_dim]
                - dones: [batch_size, 1]
        """
        # Random sampling without replacement
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
        
        return batch
    
    def load_from_dataset(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Load entire dataset into buffer (for offline RL).
        
        Args:
            states: Array of states [n_transitions, state_dim]
            actions: Array of actions [n_transitions, action_dim]
            rewards: Array of rewards [n_transitions, 1]
            next_states: Array of next states [n_transitions, state_dim]
            dones: Array of done flags [n_transitions, 1]
        """
        n_transitions = len(states)
        
        if n_transitions > self.capacity:
            logger.warning(
                f"Dataset size {n_transitions} exceeds capacity {self.capacity}. "
                f"Only loading last {self.capacity} transitions."
            )
            # Take last capacity transitions
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_states = next_states[-self.capacity:]
            dones = dones[-self.capacity:]
            n_transitions = self.capacity
        
        # Load data
        self.states[:n_transitions] = states
        self.actions[:n_transitions] = actions
        self.rewards[:n_transitions] = rewards.reshape(-1, 1)
        self.next_states[:n_transitions] = next_states
        self.dones[:n_transitions] = dones.reshape(-1, 1)
        
        self.size = n_transitions
        self.ptr = 0
        
        logger.info(f"Loaded {n_transitions} transitions into buffer")
    
    def save(self, path: str) -> None:
        """
        Save buffer to disk.
        
        Args:
            path: Path to save buffer
        """
        np.savez(
            path,
            states=self.states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
            ptr=self.ptr,
            size=self.size
        )
        logger.info(f"Saved buffer to {path}")
    
    def load(self, path: str) -> None:
        """
        Load buffer from disk.
        
        Args:
            path: Path to buffer file
        """
        data = np.load(path)
        
        self.states[:len(data['states'])] = data['states']
        self.actions[:len(data['actions'])] = data['actions']
        self.rewards[:len(data['rewards'])] = data['rewards']
        self.next_states[:len(data['next_states'])] = data['next_states']
        self.dones[:len(data['dones'])] = data['dones']
        self.ptr = int(data['ptr'])
        self.size = int(data['size'])
        
        logger.info(f"Loaded buffer from {path} ({self.size} transitions)")
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute statistics of stored transitions.
        
        Returns:
            Dictionary of statistics
        """
        if self.size == 0:
            return {}
        
        return {
            'size': self.size,
            'mean_reward': float(self.rewards[:self.size].mean()),
            'std_reward': float(self.rewards[:self.size].std()),
            'min_reward': float(self.rewards[:self.size].min()),
            'max_reward': float(self.rewards[:self.size].max()),
            'done_rate': float(self.dones[:self.size].mean())
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.
    
    Samples transitions with probability proportional to their TD error.
    Useful for focusing learning on surprising transitions.
    
    Sampling probability:
        P(i) = p_i^α / Σ_j p_j^α
    
    where p_i is the priority of transition i (typically |TD error| + ε)
    
    Reference: Schaul et al. (2015) "Prioritized Experience Replay"
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = 'cpu',
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            device: Computation device
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta per sample
            epsilon: Small constant to ensure non-zero priorities
        """
        super().__init__(capacity, state_dim, action_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Priority storage
        self.priorities = np.ones(capacity, dtype=np.float32) * epsilon
        self.max_priority = 1.0
        
        logger.info(f"Initialized PrioritizedReplayBuffer (α={alpha}, β={beta})")
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add transition with priority.
        
        Args:
            state, action, reward, next_state, done: Transition components
            priority: Optional priority value (uses max priority if None)
        """
        super().add(state, action, reward, next_state, done)
        
        # Set priority (use max if not provided)
        if priority is None:
            priority = self.max_priority
        
        self.priorities[self.ptr] = priority
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritization and importance sampling weights.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            batch: Dictionary of batched transitions
            indices: Indices of sampled transitions
            weights: Importance sampling weights
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices according to priorities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        # w_i = (N * P(i))^(-β) / max_j w_j
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max weight
        
        # Get batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
        
        # Increment beta (anneal towards 1.0)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priority values (typically |TD error|)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority)


class TrajectoryBuffer:
    """
    Buffer for storing complete trajectories (episodes).
    
    Useful for algorithms that require full episode information, such as
    Monte Carlo methods or trajectory-level importance sampling.
    """
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        """
        Initialize trajectory buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
            device: Computation device
        """
        self.capacity = capacity
        self.device = device
        self.trajectories = deque(maxlen=capacity)
        
        logger.info(f"Initialized TrajectoryBuffer with capacity {capacity}")
    
    def add_trajectory(self, trajectory: Dict[str, np.ndarray]) -> None:
        """
        Add complete trajectory.
        
        Args:
            trajectory: Dictionary containing:
                - states: [T, state_dim]
                - actions: [T, action_dim]
                - rewards: [T]
                - dones: [T]
        """
        self.trajectories.append(trajectory)
    
    def sample(self, batch_size: int) -> list:
        """
        Sample random trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
        
        Returns:
            List of trajectory dictionaries
        """
        indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        return [self.trajectories[i] for i in indices]
    
    def __len__(self) -> int:
        """Return number of stored trajectories."""
        return len(self.trajectories)
