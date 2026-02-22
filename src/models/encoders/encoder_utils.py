"""
Utility functions for patient state encoders.

This module provides common utilities used across different encoder implementations,
including masking, positional encoding, weight initialization, and more.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.
    
    Creates a boolean mask where True indicates valid positions and False
    indicates padding positions. This is used to mask out padded elements
    in attention mechanisms and loss computation.
    
    Args:
        lengths: Tensor of shape [batch_size] containing actual sequence lengths
        max_len: Maximum sequence length. If None, uses max(lengths)
    
    Returns:
        Boolean mask of shape [batch_size, max_len]
        True for valid positions, False for padding
    
    Example:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = create_padding_mask(lengths, max_len=6)
        >>> print(mask)
        tensor([[ True,  True,  True, False, False, False],
                [ True,  True,  True,  True,  True, False],
                [ True,  True, False, False, False, False]])
    """
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.size(0)
    
    # Create range tensor [0, 1, 2, ..., max_len-1]
    range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, max_len]
    
    # Expand lengths to [batch_size, 1] and compare
    lengths_expanded = lengths.unsqueeze(1)  # [batch_size, 1]
    
    # Create mask: position < length
    mask = range_tensor < lengths_expanded  # [batch_size, max_len]
    
    return mask


def create_attention_mask(
    padding_mask: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    Create attention mask from padding mask.
    
    Converts a padding mask to an attention mask suitable for multi-head attention.
    Optionally creates a causal mask for autoregressive models.
    
    Args:
        padding_mask: Boolean mask [batch_size, seq_len]
                     True for valid positions, False for padding
        causal: If True, creates causal (lower-triangular) mask
               for autoregressive attention
    
    Returns:
        Attention mask of shape [batch_size, seq_len, seq_len]
        True indicates positions that CAN attend to each other
    
    Example:
        >>> padding_mask = torch.tensor([[True, True, False]])
        >>> attn_mask = create_attention_mask(padding_mask, causal=True)
        >>> print(attn_mask[0])
        tensor([[ True, False, False],
                [ True,  True, False],
                [False, False, False]])
    """
    batch_size, seq_len = padding_mask.shape
    
    # Create basic attention mask: [batch_size, 1, seq_len] x [batch_size, seq_len, 1]
    # Result: [batch_size, seq_len, seq_len]
    attn_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
    
    if causal:
        # Create lower triangular causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=padding_mask.device, dtype=torch.bool)
        )
        # Combine with padding mask
        attn_mask = attn_mask & causal_mask.unsqueeze(0)
    
    return attn_mask


def create_positional_encoding(
    seq_len: int,
    d_model: int,
    max_len: int = 5000
) -> torch.Tensor:
    """
    Create sinusoidal positional encoding.
    
    Implements the positional encoding from "Attention is All You Need":
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        seq_len: Sequence length to encode
        d_model: Dimension of the model/encoding
        max_len: Maximum sequence length to pre-compute
    
    Returns:
        Positional encoding tensor of shape [seq_len, d_model]
    
    Example:
        >>> pe = create_positional_encoding(seq_len=10, d_model=512)
        >>> print(pe.shape)
        torch.Size([10, 512])
    """
    # Create position indices [0, 1, 2, ..., seq_len-1]
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    
    # Create dimension indices [0, 2, 4, ..., d_model-2]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * 
        -(math.log(10000.0) / d_model)
    )  # [d_model/2]
    
    # Initialize positional encoding tensor
    pe = torch.zeros(seq_len, d_model)
    
    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cosine to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that can be added to embeddings.
    
    This is a learnable or fixed positional encoding that adds temporal
    information to input embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        learnable: bool = False
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout probability
            learnable: If True, make positional encoding learnable parameters
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.learnable = learnable
        
        if learnable:
            # Learnable positional embeddings
            self.pe = nn.Parameter(torch.randn(max_len, d_model))
        else:
            # Fixed sinusoidal positional encoding
            pe = create_positional_encoding(max_len, d_model)
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Input with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        
        return self.dropout(x)


def initialize_weights(model: nn.Module, method: str = 'xavier_uniform'):
    """
    Initialize model weights using specified method.
    
    Args:
        model: PyTorch model to initialize
        method: Initialization method
               ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    
    Example:
        >>> model = nn.Linear(10, 5)
        >>> initialize_weights(model, method='xavier_uniform')
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:  # Linear/Conv layers
                if method == 'xavier_uniform':
                    init.xavier_uniform_(param)
                elif method == 'xavier_normal':
                    init.xavier_normal_(param)
                elif method == 'kaiming_uniform':
                    init.kaiming_uniform_(param, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    init.kaiming_normal_(param, nonlinearity='relu')
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
        elif 'bias' in name:
            init.constant_(param, 0)
    
    logger.info(f"Initialized model weights using {method}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    
    Example:
        >>> model = nn.Linear(100, 50)
        >>> print(count_parameters(model))
        5050  # 100*50 + 50 (bias)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_activation_function(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
              ('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu')
    
    Returns:
        PyTorch activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
    }
    
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation function: {name}. "
            f"Available: {list(activations.keys())}"
        )
    
    return activations[name.lower()]


def compute_sequence_mask_statistics(mask: torch.Tensor) -> dict:
    """
    Compute statistics about sequence masks.
    
    Args:
        mask: Boolean mask [batch_size, seq_len]
    
    Returns:
        Dictionary with mask statistics
    """
    lengths = mask.sum(dim=1).float()
    
    return {
        'mean_length': lengths.mean().item(),
        'std_length': lengths.std().item(),
        'min_length': lengths.min().item(),
        'max_length': lengths.max().item(),
        'total_valid_tokens': mask.sum().item(),
        'total_tokens': mask.numel(),
        'padding_ratio': 1.0 - (mask.sum().item() / mask.numel())
    }


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower-triangular) mask for autoregressive models.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        Boolean mask [seq_len, seq_len] where True indicates valid attention
    
    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[ True, False, False, False],
                [ True,  True, False, False],
                [ True,  True,  True, False],
                [ True,  True,  True,  True]])
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute mean of tensor along dimension, ignoring masked positions.
    
    Args:
        tensor: Input tensor [batch_size, seq_len, feature_dim]
        mask: Boolean mask [batch_size, seq_len]
        dim: Dimension to compute mean over
        keepdim: Whether to keep the reduced dimension
    
    Returns:
        Masked mean tensor
    """
    # Expand mask to match tensor dimensions
    mask_expanded = mask.unsqueeze(-1) if dim == 1 else mask
    
    # Zero out masked positions
    masked_tensor = tensor * mask_expanded.float()
    
    # Compute sum and count of valid positions
    sum_tensor = masked_tensor.sum(dim=dim, keepdim=keepdim)
    count = mask_expanded.float().sum(dim=dim, keepdim=keepdim).clamp(min=1e-9)
    
    return sum_tensor / count


def masked_max(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    keepdim: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute max of tensor along dimension, ignoring masked positions.
    
    Args:
        tensor: Input tensor [batch_size, seq_len, feature_dim]
        mask: Boolean mask [batch_size, seq_len]
        dim: Dimension to compute max over
        keepdim: Whether to keep the reduced dimension
    
    Returns:
        Tuple of (max_values, max_indices)
    """
    # Expand mask to match tensor dimensions
    mask_expanded = mask.unsqueeze(-1) if dim == 1 else mask
    
    # Set masked positions to very negative value
    masked_tensor = tensor.clone()
    masked_tensor[~mask_expanded] = float('-inf')
    
    return torch.max(masked_tensor, dim=dim, keepdim=keepdim)
