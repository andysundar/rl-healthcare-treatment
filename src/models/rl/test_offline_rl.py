"""
Test Suite for Offline RL Package

Tests for all major components to ensure correctness.

Run with: pytest tests/test_offline_rl.py -v

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.rl import (
    CQLAgent,
    BCQAgent,
    ReplayBuffer,
    OfflineRLTrainer,
    CQLConfig,
    QNetwork,
    PolicyNetwork,
    ValueNetwork
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    n_samples = 1000
    state_dim = 10
    action_dim = 3
    
    states = np.random.randn(n_samples, state_dim).astype(np.float32)
    actions = np.random.randn(n_samples, action_dim).astype(np.float32)
    rewards = np.random.randn(n_samples).astype(np.float32)
    next_states = np.random.randn(n_samples, state_dim).astype(np.float32)
    dones = (np.random.rand(n_samples) > 0.95).astype(np.float32)
    
    return states, actions, rewards, next_states, dones


class TestNetworks:
    """Test neural network architectures."""
    
    def test_q_network_forward(self):
        """Test Q-network forward pass."""
        state_dim = 10
        action_dim = 3
        batch_size = 32
        
        q_net = QNetwork(state_dim, action_dim, hidden_dim=64)
        
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        
        q_values = q_net(states, actions)
        
        assert q_values.shape == (batch_size, 1)
        assert not torch.isnan(q_values).any()
    
    def test_policy_network_continuous(self):
        """Test policy network with continuous actions."""
        state_dim = 10
        action_dim = 3
        batch_size = 32
        
        policy = PolicyNetwork(
            state_dim, 
            action_dim, 
            hidden_dim=64,
            action_space='continuous'
        )
        
        states = torch.randn(batch_size, state_dim)
        
        # Test stochastic action
        actions, log_probs = policy(states, deterministic=False)
        assert actions.shape == (batch_size, action_dim)
        assert log_probs.shape == (batch_size, 1)
        
        # Test deterministic action
        actions_det, _ = policy(states, deterministic=True)
        assert actions_det.shape == (batch_size, action_dim)
    
    def test_policy_network_discrete(self):
        """Test policy network with discrete actions."""
        state_dim = 10
        action_dim = 5
        batch_size = 32
        
        policy = PolicyNetwork(
            state_dim,
            action_dim,
            hidden_dim=64,
            action_space='discrete'
        )
        
        states = torch.randn(batch_size, state_dim)
        actions, log_probs = policy(states)
        
        assert actions.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size, 1)
    
    def test_value_network(self):
        """Test value network."""
        state_dim = 10
        batch_size = 32
        
        value_net = ValueNetwork(state_dim, hidden_dim=64)
        states = torch.randn(batch_size, state_dim)
        
        values = value_net(states)
        assert values.shape == (batch_size, 1)


class TestReplayBuffer:
    """Test replay buffer functionality."""
    
    def test_buffer_add_and_sample(self, synthetic_data):
        """Test adding transitions and sampling."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        buffer = ReplayBuffer(
            capacity=10000,
            state_dim=states.shape[1],
            action_dim=actions.shape[1]
        )
        
        # Add transitions
        for i in range(len(states)):
            buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        assert len(buffer) == len(states)
        
        # Sample batch
        batch = buffer.sample(batch_size=32)
        
        assert batch['states'].shape == (32, states.shape[1])
        assert batch['actions'].shape == (32, actions.shape[1])
        assert batch['rewards'].shape == (32, 1)
    
    def test_buffer_load_dataset(self, synthetic_data):
        """Test loading entire dataset."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        buffer = ReplayBuffer(
            capacity=10000,
            state_dim=states.shape[1],
            action_dim=actions.shape[1]
        )
        
        buffer.load_from_dataset(states, actions, rewards, next_states, dones)
        
        assert len(buffer) == len(states)
        
        # Check statistics
        stats = buffer.get_statistics()
        assert 'mean_reward' in stats
        assert 'std_reward' in stats
    
    def test_buffer_save_load(self, synthetic_data, temp_dir):
        """Test buffer save and load."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        buffer = ReplayBuffer(
            capacity=10000,
            state_dim=states.shape[1],
            action_dim=actions.shape[1]
        )
        buffer.load_from_dataset(states, actions, rewards, next_states, dones)
        
        # Save buffer
        save_path = Path(temp_dir) / 'buffer.npz'
        buffer.save(str(save_path))
        
        # Load into new buffer
        new_buffer = ReplayBuffer(
            capacity=10000,
            state_dim=states.shape[1],
            action_dim=actions.shape[1]
        )
        new_buffer.load(str(save_path))
        
        assert len(new_buffer) == len(buffer)


class TestCQLAgent:
    """Test CQL agent."""
    
    def test_cql_initialization(self):
        """Test CQL agent initialization."""
        agent = CQLAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64,
            cql_alpha=1.0
        )
        
        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert agent.cql_alpha == 1.0
    
    def test_cql_action_selection(self):
        """Test action selection."""
        agent = CQLAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64
        )
        
        state = np.random.randn(10).astype(np.float32)
        
        # Deterministic action
        action_det = agent.select_action(state, deterministic=True)
        assert action_det.shape == (3,)
        
        # Stochastic action
        action_stoch = agent.select_action(state, deterministic=False)
        assert action_stoch.shape == (3,)
    
    def test_cql_training_step(self, synthetic_data):
        """Test CQL training step."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        agent = CQLAgent(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=64
        )
        
        # Create batch
        batch = {
            'states': torch.FloatTensor(states[:32]),
            'actions': torch.FloatTensor(actions[:32]),
            'rewards': torch.FloatTensor(rewards[:32]).unsqueeze(1),
            'next_states': torch.FloatTensor(next_states[:32]),
            'dones': torch.FloatTensor(dones[:32]).unsqueeze(1)
        }
        
        # Training step
        metrics = agent.train_step(batch)
        
        assert 'q1_loss' in metrics
        assert 'q2_loss' in metrics
        assert 'cql_penalty' in metrics
        assert 'policy_loss' in metrics
        assert not np.isnan(metrics['q1_loss'])
    
    def test_cql_get_q_value(self):
        """Test Q-value computation."""
        agent = CQLAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64
        )
        
        state = np.random.randn(10).astype(np.float32)
        action = np.random.randn(3).astype(np.float32)
        
        q_value = agent.get_q_value(state, action)
        
        assert isinstance(q_value, float)
        assert not np.isnan(q_value)
    
    def test_cql_save_load(self, temp_dir):
        """Test saving and loading agent."""
        agent = CQLAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64
        )
        
        # Save agent
        save_path = Path(temp_dir) / 'agent.pt'
        agent.save(str(save_path))
        
        # Load into new agent
        new_agent = CQLAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64
        )
        new_agent.load(str(save_path))
        
        # Check same state
        state = np.random.randn(10).astype(np.float32)
        action1 = agent.select_action(state, deterministic=True)
        action2 = new_agent.select_action(state, deterministic=True)
        
        assert np.allclose(action1, action2, atol=1e-5)


class TestBCQAgent:
    """Test BCQ agent."""
    
    def test_bcq_initialization(self):
        """Test BCQ agent initialization."""
        agent = BCQAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64,
            latent_dim=32
        )
        
        assert agent.state_dim == 10
        assert agent.action_dim == 3
    
    def test_bcq_action_selection(self):
        """Test action selection."""
        agent = BCQAgent(
            state_dim=10,
            action_dim=3,
            hidden_dim=64
        )
        
        state = np.random.randn(10).astype(np.float32)
        action = agent.select_action(state)
        
        assert action.shape == (3,)
        assert not np.isnan(action).any()
    
    def test_bcq_training_step(self, synthetic_data):
        """Test BCQ training step."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        agent = BCQAgent(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=64
        )
        
        batch = {
            'states': torch.FloatTensor(states[:32]),
            'actions': torch.FloatTensor(actions[:32]),
            'rewards': torch.FloatTensor(rewards[:32]).unsqueeze(1),
            'next_states': torch.FloatTensor(next_states[:32]),
            'dones': torch.FloatTensor(dones[:32]).unsqueeze(1)
        }
        
        metrics = agent.train_step(batch)
        
        assert 'vae_loss' in metrics
        assert 'q1_loss' in metrics
        assert 'perturbation_loss' in metrics


class TestTrainer:
    """Test training functionality."""
    
    def test_trainer_initialization(self, synthetic_data):
        """Test trainer initialization."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        agent = CQLAgent(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=64
        )
        
        buffer = ReplayBuffer(
            capacity=10000,
            state_dim=states.shape[1],
            action_dim=actions.shape[1]
        )
        buffer.load_from_dataset(states, actions, rewards, next_states, dones)
        
        trainer = OfflineRLTrainer(
            agent=agent,
            replay_buffer=buffer,
            save_dir='./test_checkpoints'
        )
        
        assert trainer.agent == agent
        assert trainer.replay_buffer == buffer
    
    def test_trainer_short_training(self, synthetic_data, temp_dir):
        """Test short training run."""
        states, actions, rewards, next_states, dones = synthetic_data
        
        agent = CQLAgent(
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=64
        )
        
        buffer = ReplayBuffer(
            capacity=10000,
            state_dim=states.shape[1],
            action_dim=actions.shape[1]
        )
        buffer.load_from_dataset(states, actions, rewards, next_states, dones)
        
        trainer = OfflineRLTrainer(
            agent=agent,
            replay_buffer=buffer,
            save_dir=temp_dir,
            eval_freq=100,
            save_freq=500
        )
        
        # Short training
        history = trainer.train(
            num_iterations=100,
            batch_size=32,
            verbose=False
        )
        
        assert len(history['train_losses']) > 0


class TestConfiguration:
    """Test configuration management."""
    
    def test_cql_config(self):
        """Test CQL configuration."""
        config = CQLConfig(
            state_dim=10,
            action_dim=3,
            hidden_dim=128,
            cql_alpha=2.0
        )
        
        assert config.state_dim == 10
        assert config.cql_alpha == 2.0
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict['state_dim'] == 10
        
        # Test from_dict
        new_config = CQLConfig.from_dict(config_dict)
        assert new_config.state_dim == 10
    
    def test_config_save_load(self, temp_dir):
        """Test configuration save and load."""
        config = CQLConfig(
            state_dim=10,
            action_dim=3
        )
        
        save_path = Path(temp_dir) / 'config.yaml'
        config.save(str(save_path))
        
        loaded_config = CQLConfig.from_yaml(str(save_path))
        assert loaded_config.state_dim == config.state_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
