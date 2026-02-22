"""
Offline RL Trainer

This module provides training infrastructure for offline RL agents, including:
- Training loop with gradient updates
- Periodic evaluation
- Checkpointing and model persistence
- Metrics logging (MLflow, Weights & Biases)
- Early stopping

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from tqdm import tqdm
import logging
import json
from datetime import datetime

from .base_agent import BaseRLAgent
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class OfflineRLTrainer:
    """
    Trainer for offline reinforcement learning algorithms.
    
    Handles the complete training pipeline:
    1. Load dataset into replay buffer
    2. Training iterations with batch sampling
    3. Periodic evaluation on validation set
    4. Checkpoint saving
    5. Metrics logging
    6. Early stopping based on evaluation performance
    """
    
    def __init__(
        self,
        agent: BaseRLAgent,
        replay_buffer: ReplayBuffer,
        save_dir: str = './checkpoints',
        eval_freq: int = 1000,
        save_freq: int = 5000,
        log_freq: int = 100,
        use_mlflow: bool = False,
        use_wandb: bool = False,
        early_stopping_patience: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            agent: RL agent to train
            replay_buffer: Replay buffer containing offline dataset
            save_dir: Directory for saving checkpoints
            eval_freq: Frequency of evaluation (in training steps)
            save_freq: Frequency of checkpoint saving
            log_freq: Frequency of metric logging
            use_mlflow: Whether to use MLflow for logging
            use_wandb: Whether to use Weights & Biases for logging
            early_stopping_patience: Number of evaluations without improvement before stopping
        """
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_eval_return = -np.inf
        self.patience_counter = 0
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize loggers
        self._init_loggers()
        
        logger.info(f"Initialized OfflineRLTrainer")
        logger.info(f"  Save directory: {save_dir}")
        logger.info(f"  Eval frequency: {eval_freq}")
        logger.info(f"  Replay buffer size: {len(replay_buffer)}")
    
    def _init_loggers(self):
        """Initialize external logging frameworks."""
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_experiment("offline_rl_healthcare")
                logger.info("Initialized MLflow logging")
            except ImportError:
                logger.warning("MLflow not installed, disabling MLflow logging")
                self.use_mlflow = False
        
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project="healthcare-offline-rl")
                logger.info("Initialized Weights & Biases logging")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
    
    def train(
        self,
        num_iterations: int,
        batch_size: int = 256,
        eval_fn: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            batch_size: Batch size for training
            eval_fn: Optional evaluation function that returns metrics
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training for {num_iterations} iterations")
        logger.info(f"Batch size: {batch_size}")
        
        history = {
            'train_losses': [],
            'eval_returns': [],
            'eval_steps': []
        }
        
        # Set agent to training mode
        self.agent.train_mode()
        
        # Training loop
        pbar = tqdm(range(num_iterations), disable=not verbose, desc="Training")
        
        for iteration in pbar:
            # Sample batch
            batch = self.replay_buffer.sample(batch_size)
            
            # Training step
            metrics = self.agent.train_step(batch)
            
            self.global_step += 1
            
            # Log metrics
            if self.global_step % self.log_freq == 0:
                self._log_metrics(metrics, prefix='train')
                history['train_losses'].append(metrics)
                
                if verbose:
                    pbar.set_postfix({
                        k: f"{v:.4f}" for k, v in metrics.items() 
                        if isinstance(v, (int, float))
                    })
            
            # Evaluation
            if self.global_step % self.eval_freq == 0 and eval_fn is not None:
                eval_metrics = self._evaluate(eval_fn)
                history['eval_returns'].append(eval_metrics['mean_return'])
                history['eval_steps'].append(self.global_step)
                
                # Early stopping check
                if self._check_early_stopping(eval_metrics['mean_return']):
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
            
            # Save checkpoint
            if self.global_step % self.save_freq == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        logger.info("Training completed")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        return history
    
    def _evaluate(self, eval_fn: Callable) -> Dict[str, float]:
        """
        Run evaluation.
        
        Args:
            eval_fn: Function that evaluates the agent and returns metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Running evaluation at step {self.global_step}")
        
        # Set agent to evaluation mode
        self.agent.eval_mode()
        
        # Run evaluation
        eval_metrics = eval_fn(self.agent)
        
        # Log evaluation metrics
        self._log_metrics(eval_metrics, prefix='eval')
        
        logger.info(f"Evaluation: {eval_metrics}")
        
        # Set agent back to training mode
        self.agent.train_mode()
        
        return eval_metrics
    
    def _check_early_stopping(self, eval_return: float) -> bool:
        """
        Check early stopping condition.
        
        Args:
            eval_return: Current evaluation return
        
        Returns:
            True if should stop training
        """
        if eval_return > self.best_eval_return:
            self.best_eval_return = eval_return
            self.patience_counter = 0
            
            # Save best model
            self.save_checkpoint('best_model.pt')
            logger.info(f"New best model: {self.best_eval_return:.4f}")
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stopping_patience:
            return True
        
        return False
    
    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ''):
        """
        Log metrics to various logging backends.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names (e.g., 'train', 'eval')
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to MLflow
        if self.use_mlflow:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(key, value, step=self.global_step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.wandb.log(wandb_metrics, step=self.global_step)
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_return': self.best_eval_return,
            'patience_counter': self.patience_counter,
            'agent_state': self.agent.get_config()
        }
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save training state
        state_path = checkpoint_path.parent / f"{checkpoint_path.stem}_training_state.json"
        with open(state_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.save_dir / filename
        
        # Load agent
        self.agent.load(str(checkpoint_path))
        
        # Load training state
        state_path = checkpoint_path.parent / f"{checkpoint_path.stem}_training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            self.best_eval_return = checkpoint['best_eval_return']
            self.patience_counter = checkpoint['patience_counter']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.
        
        Returns:
            Dictionary of training summary statistics
        """
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_return': self.best_eval_return,
            'patience_counter': self.patience_counter,
            'buffer_size': len(self.replay_buffer),
            'agent_config': self.agent.get_config()
        }


class EvaluationManager:
    """
    Manager for evaluating offline RL policies.
    
    Provides various evaluation methods:
    - On-policy evaluation (if environment available)
    - Off-policy evaluation (using historical data)
    - Safety metrics
    """
    
    def __init__(
        self,
        agent: BaseRLAgent,
        eval_buffer: Optional[ReplayBuffer] = None
    ):
        """
        Initialize evaluation manager.
        
        Args:
            agent: Agent to evaluate
            eval_buffer: Optional buffer with evaluation data
        """
        self.agent = agent
        self.eval_buffer = eval_buffer
    
    def evaluate_on_episodes(
        self,
        eval_episodes: list,
        max_episode_length: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate agent on pre-collected episodes.
        
        Args:
            eval_episodes: List of episode dictionaries
            max_episode_length: Maximum episode length
        
        Returns:
            Evaluation metrics
        """
        self.agent.eval_mode()
        
        returns = []
        episode_lengths = []
        
        for episode in eval_episodes:
            states = episode['states']
            episode_return = 0.0
            
            for t, state in enumerate(states):
                action = self.agent.select_action(state, deterministic=True)
                reward = episode['rewards'][t]
                episode_return += reward
                
                if t >= max_episode_length:
                    break
            
            returns.append(episode_return)
            episode_lengths.append(len(states))
        
        metrics = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_episode_length': np.mean(episode_lengths)
        }
        
        return metrics
    
    def compute_q_value_statistics(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics of Q-values.
        
        Args:
            states: Array of states [n_samples, state_dim]
            actions: Array of actions [n_samples, action_dim]
        
        Returns:
            Q-value statistics
        """
        self.agent.eval_mode()
        
        q_values = []
        for state, action in zip(states, actions):
            q_value = self.agent.get_q_value(state, action)
            q_values.append(q_value)
        
        q_values = np.array(q_values)
        
        return {
            'mean_q_value': float(np.mean(q_values)),
            'std_q_value': float(np.std(q_values)),
            'min_q_value': float(np.min(q_values)),
            'max_q_value': float(np.max(q_values))
        }


def create_simple_eval_function(
    eval_episodes: list,
    max_episode_length: int = 1000
) -> Callable:
    """
    Create simple evaluation function for trainer.
    
    Args:
        eval_episodes: List of evaluation episodes
        max_episode_length: Maximum episode length
    
    Returns:
        Evaluation function
    """
    def eval_fn(agent: BaseRLAgent) -> Dict[str, float]:
        evaluator = EvaluationManager(agent)
        return evaluator.evaluate_on_episodes(eval_episodes, max_episode_length)
    
    return eval_fn
