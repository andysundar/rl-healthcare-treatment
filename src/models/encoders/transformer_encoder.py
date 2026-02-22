"""
Transformer-based patient state encoder.

This module implements a Transformer encoder specifically designed for patient
sequential data, with multi-modal input handling and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from base_encoder import BasePatientEncoder
from encoder_config import EncoderConfig
from encoder_utils import (
    create_padding_mask,
    PositionalEncoding,
    initialize_weights,
    masked_mean,
    masked_max
)


logger = logging.getLogger(__name__)


class TransformerPatientEncoder(BasePatientEncoder):
    """
    Transformer-based encoder for patient sequential data.
    
    This encoder uses multi-head self-attention to process sequential patient
    data (labs, vitals, medications, etc.) and produces fixed-size state
    embeddings. It handles multiple modalities and variable-length sequences.
    
    Architecture:
        1. Input embedding layers for each modality (labs, vitals, meds, etc.)
        2. Positional encoding for temporal information
        3. Multi-layer Transformer encoder with self-attention
        4. Pooling layer (mean/max/last) to get fixed-size output
        5. Output projection to state embedding dimension
    
    The encoder supports:
    - Variable-length sequences with proper masking
    - Multi-modal inputs (continuous and categorical features)
    - Attention weight extraction for interpretability
    - Pre-training on reconstruction tasks
    
    Args:
        config: Encoder configuration
    
    Example:
        >>> config = EncoderConfig(
        ...     encoder_type='transformer',
        ...     hidden_dim=256,
        ...     state_dim=128,
        ...     num_layers=4,
        ...     num_heads=8
        ... )
        >>> encoder = TransformerPatientEncoder(config)
        >>> 
        >>> # Create sample batch
        >>> batch = {
        ...     'labs': torch.randn(32, 50, 20),  # [batch, seq_len, lab_dim]
        ...     'vitals': torch.randn(32, 50, 10),
        ...     'demographics': torch.randn(32, 8),
        ...     'medications': torch.randint(0, 500, (32, 50)),
        ...     'seq_lengths': torch.randint(10, 50, (32,))
        ... }
        >>> 
        >>> embeddings = encoder(batch)  # [32, 128]
    """
    
    def __init__(self, config: EncoderConfig):
        """Initialize Transformer patient encoder."""
        super().__init__(config)
        
        # Input embedding layers for different modalities
        self.lab_encoder = nn.Linear(config.lab_dim, config.hidden_dim)
        self.vital_encoder = nn.Linear(config.vital_dim, config.hidden_dim)
        self.demo_encoder = nn.Linear(config.demo_dim, config.hidden_dim)
        
        # Embedding layers for categorical features
        self.med_embedding = nn.Embedding(
            config.med_vocab_size, 
            config.hidden_dim,
            padding_idx=0  # Assume 0 is padding
        )
        self.diag_embedding = nn.Embedding(
            config.diag_vocab_size,
            config.hidden_dim,
            padding_idx=0
        )
        self.proc_embedding = nn.Embedding(
            config.proc_vocab_size,
            config.hidden_dim,
            padding_idx=0
        )
        
        # Positional encoding
        if config.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=config.hidden_dim,
                max_len=config.max_seq_len,
                dropout=config.dropout
            )
        else:
            self.positional_encoding = None
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,  # Input shape: [batch, seq, feature]
            norm_first=True    # Pre-LayerNorm for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim)
        )
        
        # Pooling strategy
        self.pooling_type = 'mean'  # Options: 'mean', 'max', 'last', 'attention'
        
        # If using attention pooling
        if self.pooling_type == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(config.hidden_dim // 2, 1)
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.state_dim)
        )
        
        # Layer normalization for final output
        self.output_norm = nn.LayerNorm(config.state_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Store attention weights for interpretability
        self._attention_weights = None
        
        logger.info(
            f"Initialized TransformerPatientEncoder: "
            f"{self.count_parameters():,} parameters, "
            f"{config.num_layers} layers, {config.num_heads} heads"
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.med_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.diag_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.proc_embedding.weight, mean=0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _encode_modalities(
        self,
        patient_sequence: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode different modalities into unified representation.
        
        Args:
            patient_sequence: Dictionary of patient data tensors
        
        Returns:
            Combined embeddings [batch_size, seq_len, hidden_dim]
        """
        embeddings_list = []
        
        # Encode labs if present
        if 'labs' in patient_sequence:
            labs = patient_sequence['labs']  # [batch, seq_len, lab_dim]
            lab_emb = self.lab_encoder(labs)  # [batch, seq_len, hidden_dim]
            embeddings_list.append(lab_emb)
        
        # Encode vitals if present
        if 'vitals' in patient_sequence:
            vitals = patient_sequence['vitals']  # [batch, seq_len, vital_dim]
            vital_emb = self.vital_encoder(vitals)  # [batch, seq_len, hidden_dim]
            embeddings_list.append(vital_emb)
        
        # Encode medications if present
        if 'medications' in patient_sequence:
            meds = patient_sequence['medications']  # [batch, seq_len]
            med_emb = self.med_embedding(meds)  # [batch, seq_len, hidden_dim]
            embeddings_list.append(med_emb)
        
        # Encode diagnoses if present
        if 'diagnoses' in patient_sequence:
            diags = patient_sequence['diagnoses']  # [batch, seq_len]
            diag_emb = self.diag_embedding(diags)  # [batch, seq_len, hidden_dim]
            embeddings_list.append(diag_emb)
        
        # Encode procedures if present
        if 'procedures' in patient_sequence:
            procs = patient_sequence['procedures']  # [batch, seq_len]
            proc_emb = self.proc_embedding(procs)  # [batch, seq_len, hidden_dim]
            embeddings_list.append(proc_emb)
        
        # Combine embeddings (sum or concatenate then project)
        if len(embeddings_list) == 0:
            raise ValueError("No valid modalities found in patient_sequence")
        
        # Sum embeddings from different modalities
        combined = sum(embeddings_list)  # [batch, seq_len, hidden_dim]
        
        # Add demographics (broadcast across sequence)
        if 'demographics' in patient_sequence:
            demo = patient_sequence['demographics']  # [batch, demo_dim]
            demo_emb = self.demo_encoder(demo)  # [batch, hidden_dim]
            demo_emb = demo_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
            combined = combined + demo_emb  # Broadcasting
        
        return combined
    
    def _pool_sequence(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation.
        
        Args:
            sequence: Encoded sequence [batch, seq_len, hidden_dim]
            mask: Padding mask [batch, seq_len]
        
        Returns:
            Pooled representation [batch, hidden_dim]
        """
        if self.pooling_type == 'mean':
            if mask is not None:
                pooled = masked_mean(sequence, mask, dim=1)
            else:
                pooled = sequence.mean(dim=1)
        
        elif self.pooling_type == 'max':
            if mask is not None:
                pooled, _ = masked_max(sequence, mask, dim=1)
            else:
                pooled, _ = sequence.max(dim=1)
        
        elif self.pooling_type == 'last':
            if mask is not None:
                # Get last valid position for each sequence
                lengths = mask.sum(dim=1) - 1  # [batch]
                batch_indices = torch.arange(sequence.size(0), device=sequence.device)
                pooled = sequence[batch_indices, lengths]
            else:
                pooled = sequence[:, -1, :]  # Last position
        
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attn_scores = self.attention_pooling(sequence)  # [batch, seq_len, 1]
            
            if mask is not None:
                # Mask attention scores
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
            pooled = (sequence * attn_weights).sum(dim=1)  # [batch, hidden_dim]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return pooled
    
    def forward(
        self,
        patient_sequence: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            patient_sequence: Dictionary containing:
                - 'labs': Lab values [batch, seq_len, lab_dim]
                - 'vitals': Vital signs [batch, seq_len, vital_dim]
                - 'demographics': Demographics [batch, demo_dim]
                - 'medications': Med indices [batch, seq_len]
                - 'diagnoses': Diagnosis indices [batch, seq_len]
                - 'procedures': Procedure indices [batch, seq_len]
                - 'seq_lengths': Actual lengths [batch] (optional)
            mask: Padding mask [batch, seq_len] (optional)
        
        Returns:
            State embeddings [batch, state_dim]
        """
        # Create padding mask if not provided
        if mask is None and 'seq_lengths' in patient_sequence:
            seq_lengths = patient_sequence['seq_lengths']
            max_len = max(
                patient_sequence.get('labs', torch.zeros(1, 1, 1)).size(1),
                patient_sequence.get('vitals', torch.zeros(1, 1, 1)).size(1)
            )
            mask = create_padding_mask(seq_lengths, max_len=max_len)
            mask = mask.to(self.device)
        
        # Encode modalities
        combined_emb = self._encode_modalities(patient_sequence)  # [batch, seq, hidden]
        
        # Add positional encoding
        if self.positional_encoding is not None:
            combined_emb = self.positional_encoding(combined_emb)
        
        # Create attention mask for transformer (PyTorch expects inverted mask)
        # True -> not masked, False -> masked
        if mask is not None:
            # Transformer expects: False = attend, True = ignore
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        
        # Pass through transformer encoder
        encoded_sequence = self.transformer_encoder(
            combined_emb,
            src_key_padding_mask=src_key_padding_mask
        )  # [batch, seq, hidden]
        
        # Pool sequence to fixed size
        pooled = self._pool_sequence(encoded_sequence, mask)  # [batch, hidden]
        
        # Project to state dimension
        state_embedding = self.output_projection(pooled)  # [batch, state_dim]
        
        # Normalize output
        state_embedding = self.output_norm(state_embedding)
        
        return state_embedding
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from last forward pass.
        
        Returns:
            Attention weights if stored, else None
        """
        return self._attention_weights
    
    def set_pooling_type(self, pooling_type: str):
        """
        Change pooling strategy.
        
        Args:
            pooling_type: 'mean', 'max', 'last', or 'attention'
        """
        valid_types = ['mean', 'max', 'last', 'attention']
        if pooling_type not in valid_types:
            raise ValueError(
                f"Invalid pooling_type: {pooling_type}. "
                f"Must be one of {valid_types}"
            )
        
        self.pooling_type = pooling_type
        logger.info(f"Set pooling type to: {pooling_type}")
