"""
Base abstract class for patient state encoders.

This module defines the interface that all patient encoders must implement,
ensuring consistency across different encoder architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

from .encoder_config import EncoderConfig


logger = logging.getLogger(__name__)


class BasePatientEncoder(nn.Module, ABC):
    """
    Abstract base class for patient state encoders.
    
    All encoder implementations must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across different
    encoder architectures (Transformer, Autoencoder, BERT, etc.).
    
    The encoder transforms sequential patient data (labs, vitals, medications, etc.)
    into fixed-size state embeddings suitable for reinforcement learning agents.
    
    Attributes:
        config: Encoder configuration
        device: Device for computation ('cuda', 'mps', or 'cpu')
    """
    
    def __init__(self, config: EncoderConfig):
        """
        Initialize base patient encoder.
        
        Args:
            config: Encoder configuration object
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.to(self.device)
        
        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    @abstractmethod
    def forward(
        self, 
        patient_sequence: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            patient_sequence: Dictionary containing patient data tensors:
                - 'labs': Lab test results [batch_size, seq_len, lab_dim]
                - 'vitals': Vital signs [batch_size, seq_len, vital_dim]
                - 'demographics': Patient demographics [batch_size, demo_dim]
                - 'medications': Medication indices [batch_size, seq_len]
                - 'diagnoses': Diagnosis code indices [batch_size, seq_len]
                - 'procedures': Procedure code indices [batch_size, seq_len]
            mask: Optional attention mask [batch_size, seq_len]
                  True for valid positions, False for padding
        
        Returns:
            State embeddings [batch_size, state_dim]
        """
        pass
    
    def encode_batch(
        self, 
        batch: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a batch of patient sequences.
        
        Args:
            batch: Batch of patient data (same format as forward())
            return_attention: Whether to return attention weights (if applicable)
        
        Returns:
            State embeddings, or (embeddings, attention_weights) if return_attention=True
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        with torch.no_grad():
            embeddings = self.forward(batch)
        
        if return_attention and hasattr(self, 'get_attention_weights'):
            attention_weights = self.get_attention_weights()
            return embeddings, attention_weights
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """
        Get the output embedding dimension.
        
        Returns:
            State embedding dimension
        """
        return self.config.state_dim
    
    def save_checkpoint(self, path: Union[str, Path], **kwargs):
        """
        Save model checkpoint with configuration and optional metadata.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata to save (e.g., optimizer state, epoch)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'encoder_type': self.__class__.__name__,
            **kwargs
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Path], strict: bool = True):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce that the keys in state_dict
                   match the keys returned by this module's state_dict()
        
        Returns:
            Dictionary containing additional saved metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Verify encoder type matches
        if checkpoint['encoder_type'] != self.__class__.__name__:
            logger.warning(
                f"Loading checkpoint from {checkpoint['encoder_type']} "
                f"into {self.__class__.__name__}"
            )
        
        logger.info(f"Loaded checkpoint from {path}")
        
        # Return additional metadata
        return {k: v for k, v in checkpoint.items() 
                if k not in ['model_state_dict', 'config', 'encoder_type']}
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def freeze(self):
        """Freeze all model parameters (disable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"Froze all parameters in {self.__class__.__name__}")
    
    def unfreeze(self):
        """Unfreeze all model parameters (enable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze all parameters in {self.__class__.__name__}")
    
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return self.device
    
    def summary(self) -> Dict:
        """
        Get model summary with architecture details.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'encoder_type': self.__class__.__name__,
            'total_parameters': self.count_parameters(trainable_only=False),
            'trainable_parameters': self.count_parameters(trainable_only=True),
            'embedding_dim': self.get_embedding_dim(),
            'device': str(self.device),
            'config': self.config.to_dict()
        }
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        trainable_params = self.count_parameters(trainable_only=True)
        total_params = self.count_parameters(trainable_only=False)
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  embedding_dim={self.get_embedding_dim()},\n"
            f"  trainable_params={trainable_params:,},\n"
            f"  total_params={total_params:,},\n"
            f"  device={self.device}\n"
            f")"
        )
