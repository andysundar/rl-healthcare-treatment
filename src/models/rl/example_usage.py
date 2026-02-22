"""
Example Usage: Training CQL on Healthcare Data

This script demonstrates complete usage of the offline RL package for
healthcare treatment recommendations.

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import numpy as np
import torch
from pathlib import Path

from src.models.rl.cql import CQLAgent
from src.models.rl.bcq import BCQAgent
from src.models.rl.replay_buffer import ReplayBuffer
from src.models.rl.trainer import OfflineRLTrainer, create_simple_eval_function
from src.models.rl.config import CQLConfig, TrainingConfig, get_diabetes_management_config


def load_mimic_data(data_path: str) -> tuple:
    """
    Load MIMIC-III data.
    
    Replace this with your actual MIMIC data loading logic.
    
    Returns:
        train_data: Tuple of (states, actions, rewards, next_states, dones)
        eval_episodes: List of evaluation episodes
    """
    # Placeholder: Load your preprocessed MIMIC data
    print(f"Loading data from {data_path}...")
    
    # Example: Load from numpy arrays
    # states = np.load(f"{data_path}/states.npy")
    # actions = np.load(f"{data_path}/actions.npy")
    # rewards = np.load(f"{data_path}/rewards.npy")
    # next_states = np.load(f"{data_path}/next_states.npy")
    # dones = np.load(f"{data_path}/dones.npy")
    
    # For demonstration, create synthetic data
    n_samples = 10000
    state_dim = 128
    action_dim = 5
    
    states = np.random.randn(n_samples, state_dim).astype(np.float32)
    actions = np.random.randn(n_samples, action_dim).astype(np.float32)
    rewards = np.random.randn(n_samples).astype(np.float32)
    next_states = np.random.randn(n_samples, state_dim).astype(np.float32)
    dones = (np.random.rand(n_samples) > 0.95).astype(np.float32)
    
    train_data = (states, actions, rewards, next_states, dones)
    
    # Create evaluation episodes
    eval_episodes = []
    for i in range(10):
        episode_length = np.random.randint(50, 200)
        episode = {
            'states': np.random.randn(episode_length, state_dim),
            'actions': np.random.randn(episode_length, action_dim),
            'rewards': np.random.randn(episode_length)
        }
        eval_episodes.append(episode)
    
    return train_data, eval_episodes


def example_1_basic_cql_training():
    """Example 1: Basic CQL training with manual configuration."""
    print("\n" + "="*60)
    print("Example 1: Basic CQL Training")
    print("="*60)
    
    # Configure agent
    config = CQLConfig(
        state_dim=128,
        action_dim=5,
        hidden_dim=256,
        cql_alpha=1.0,
        gamma=0.99,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize agent
    agent = CQLAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        cql_alpha=config.cql_alpha,
        gamma=config.gamma,
        device=config.device
    )
    
    print(f"Initialized CQL agent on {config.device}")
    
    # Load data
    train_data, eval_episodes = load_mimic_data("./data/mimic")
    states, actions, rewards, next_states, dones = train_data
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=100000,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        device=config.device
    )
    
    # Load data into buffer
    replay_buffer.load_from_dataset(states, actions, rewards, next_states, dones)
    print(f"Loaded {len(replay_buffer)} transitions into buffer")
    print(f"Buffer statistics: {replay_buffer.get_statistics()}")
    
    # Create trainer
    trainer = OfflineRLTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        save_dir='./checkpoints/example1',
        eval_freq=1000,
        save_freq=5000,
        log_freq=100
    )
    
    # Create evaluation function
    eval_fn = create_simple_eval_function(eval_episodes)
    
    # Train
    history = trainer.train(
        num_iterations=10000,
        batch_size=256,
        eval_fn=eval_fn,
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Final training summary: {trainer.get_training_summary()}")


def example_2_using_config_file():
    """Example 2: Training using configuration from YAML file."""
    print("\n" + "="*60)
    print("Example 2: Training with Config File")
    print("="*60)
    
    # Load configuration
    config = get_diabetes_management_config()
    
    # Save config for reference
    config.save('./configs/diabetes_management.yaml')
    print(f"Saved config to ./configs/diabetes_management.yaml")
    
    # Initialize components from config
    agent = CQLAgent(
        state_dim=config.agent_config.state_dim,
        action_dim=config.agent_config.action_dim,
        hidden_dim=config.agent_config.hidden_dim,
        cql_alpha=config.agent_config.cql_alpha,
        gamma=config.agent_config.gamma,
        device=config.agent_config.device
    )
    
    # Load data
    train_data, eval_episodes = load_mimic_data(config.data_dir)
    states, actions, rewards, next_states, dones = train_data
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.buffer_config.capacity,
        state_dim=config.buffer_config.state_dim,
        action_dim=config.buffer_config.action_dim,
        device=config.buffer_config.device
    )
    replay_buffer.load_from_dataset(states, actions, rewards, next_states, dones)
    
    # Create trainer
    trainer = OfflineRLTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        save_dir=config.training_config.save_dir,
        eval_freq=config.training_config.eval_freq,
        save_freq=config.training_config.save_freq,
        log_freq=config.training_config.log_freq,
        early_stopping_patience=config.training_config.early_stopping_patience
    )
    
    # Train
    eval_fn = create_simple_eval_function(eval_episodes)
    history = trainer.train(
        num_iterations=config.training_config.num_iterations,
        batch_size=config.training_config.batch_size,
        eval_fn=eval_fn
    )
    
    print(f"\nTraining completed!")


def example_3_bcq_training():
    """Example 3: Training BCQ agent."""
    print("\n" + "="*60)
    print("Example 3: BCQ Training")
    print("="*60)
    
    # Initialize BCQ agent
    agent = BCQAgent(
        state_dim=128,
        action_dim=5,
        hidden_dim=256,
        latent_dim=64,
        gamma=0.99,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Initialized BCQ agent")
    
    # Load data
    train_data, eval_episodes = load_mimic_data("./data/mimic")
    states, actions, rewards, next_states, dones = train_data
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=100000,
        state_dim=128,
        action_dim=5,
        device=agent.device
    )
    replay_buffer.load_from_dataset(states, actions, rewards, next_states, dones)
    
    # Create trainer
    trainer = OfflineRLTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        save_dir='./checkpoints/example3_bcq'
    )
    
    # Train
    eval_fn = create_simple_eval_function(eval_episodes)
    history = trainer.train(
        num_iterations=10000,
        batch_size=256,
        eval_fn=eval_fn
    )
    
    print(f"\nBCQ training completed!")


def example_4_loading_and_inference():
    """Example 4: Load trained model and perform inference."""
    print("\n" + "="*60)
    print("Example 4: Loading Model and Inference")
    print("="*60)
    
    # Initialize agent with same config as trained model
    agent = CQLAgent(
        state_dim=128,
        action_dim=5,
        hidden_dim=256,
        device='cpu'
    )
    
    # Load trained model
    checkpoint_path = './checkpoints/example1/final_model.pt'
    if Path(checkpoint_path).exists():
        agent.load(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
        
        # Set to evaluation mode
        agent.eval_mode()
        
        # Perform inference on new patient states
        test_states = np.random.randn(10, 128).astype(np.float32)
        
        print("\nPerforming inference on 10 test states:")
        for i, state in enumerate(test_states):
            action = agent.select_action(state, deterministic=True)
            q_value = agent.get_q_value(state, action)
            
            print(f"  State {i}: Action = {action[:3]}..., Q-value = {q_value:.4f}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run example 1 first to train a model!")


def example_5_batch_evaluation():
    """Example 5: Batch evaluation and Q-value analysis."""
    print("\n" + "="*60)
    print("Example 5: Batch Evaluation")
    print("="*60)
    
    # Initialize and load agent
    agent = CQLAgent(state_dim=128, action_dim=5, hidden_dim=256)
    
    checkpoint_path = './checkpoints/example1/best_model.pt'
    if Path(checkpoint_path).exists():
        agent.load(checkpoint_path)
        agent.eval_mode()
        
        # Load evaluation data
        _, eval_episodes = load_mimic_data("./data/mimic")
        
        # Evaluate on episodes
        from src.models.rl.trainer import EvaluationManager
        
        evaluator = EvaluationManager(agent)
        metrics = evaluator.evaluate_on_episodes(eval_episodes)
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Compute Q-value statistics
        states = eval_episodes[0]['states'][:100]
        actions = eval_episodes[0]['actions'][:100]
        
        q_stats = evaluator.compute_q_value_statistics(states, actions)
        print("\nQ-value Statistics:")
        for key, value in q_stats.items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")


if __name__ == "__main__":
    print("Offline RL for Healthcare Treatment Recommendations")
    print("Examples demonstrating package usage\n")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run examples
    try:
        example_1_basic_cql_training()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_using_config_file()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_bcq_training()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_loading_and_inference()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_batch_evaluation()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
