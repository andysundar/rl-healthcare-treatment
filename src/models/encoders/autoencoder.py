"""
Autoencoder-based patient state encoder.

This module implements a standard autoencoder for patient data representation
learning. The encoder can be pre-trained on reconstruction tasks and then used
to generate fixed-size state embeddings for RL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .base_encoder import BasePatientEncoder
from .encoder_config import EncoderConfig
from .encoder_utils import initialize_weights


logger = logging.getLogger(__name__)


class PatientAutoencoder(BasePatientEncoder):
    """
    Autoencoder for patient state representation learning.
    
    This autoencoder compresses patient data into a low-dimensional latent
    representation and reconstructs it. The encoder part can be used to
    generate state embeddings for RL agents.
    
    Architecture:
        Encoder: input_dim -> 512 -> 256 -> latent_dim (state_dim)
        Decoder: latent_dim -> 256 -> 512 -> input_dim
    
    Features:
    - Bottleneck architecture for dimensionality reduction
    - Dropout for regularization
    - Optional variational autoencoder (VAE) mode
    - Pre-training on reconstruction task
    
    Args:
        config: Encoder configuration
        variational: If True, use VAE with KL divergence regularization
    
    Example:
        >>> config = EncoderConfig(
        ...     encoder_type='autoencoder',
        ...     state_dim=128,
        ...     lab_dim=20,
        ...     vital_dim=10,
        ...     demo_dim=8
        ... )
        >>> autoencoder = PatientAutoencoder(config)
        >>> 
        >>> # Flatten patient data
        >>> batch_data = torch.randn(32, 38)  # lab_dim + vital_dim + demo_dim
        >>> 
        >>> # Training mode: get reconstruction and latent
        >>> reconstruction, latent = autoencoder.forward(batch_data)
        >>> loss = F.mse_loss(reconstruction, batch_data)
        >>> 
        >>> # Inference mode: just encode
        >>> latent = autoencoder.encode(batch_data)
    """
    
    def __init__(
        self,
        config: EncoderConfig,
        variational: bool = False
    ):
        """Initialize autoencoder."""
        super().__init__(config)
        
        self.variational = variational
        
        # Calculate total input dimension
        self.input_dim = config.get_total_input_dim()
        
        # Encoder architecture
        encoder_layers = []
        
        # Input layer
        encoder_layers.extend([
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        # Hidden layer
        encoder_layers.extend([
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        self.encoder_base = nn.Sequential(*encoder_layers)
        
        # Latent layer
        if variational:
            # VAE: separate layers for mean and log variance
            self.fc_mu = nn.Linear(256, config.state_dim)
            self.fc_logvar = nn.Linear(256, config.state_dim)
        else:
            # Standard AE: single latent layer
            self.fc_latent = nn.Linear(256, config.state_dim)
        
        # Decoder architecture
        decoder_layers = []
        
        # From latent to hidden
        decoder_layers.extend([
            nn.Linear(config.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        # Hidden layer
        decoder_layers.extend([
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        # Output layer
        decoder_layers.append(nn.Linear(512, self.input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"Initialized {'Variational ' if variational else ''}PatientAutoencoder: "
            f"input_dim={self.input_dim}, latent_dim={config.state_dim}, "
            f"parameters={self.count_parameters():,}"
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _flatten_patient_data(
        self,
        patient_sequence: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Flatten patient data dictionary to vector.
        
        For sequential data, we take the mean across time steps.
        For static data (demographics), we use as-is.
        
        Args:
            patient_sequence: Dictionary of patient data
        
        Returns:
            Flattened data [batch, input_dim]
        """
        features = []
        
        # Labs: take mean if sequential, otherwise use directly
        if 'labs' in patient_sequence:
            labs = patient_sequence['labs']
            if labs.dim() == 3:  # [batch, seq_len, lab_dim]
                labs = labs.mean(dim=1)  # [batch, lab_dim]
            features.append(labs)
        else:
            # If labs not present, use zeros
            batch_size = patient_sequence.get('demographics', torch.zeros(1, self.config.demo_dim)).size(0)
            features.append(torch.zeros(batch_size, self.config.lab_dim, device=self.device))
        
        # Vitals: take mean if sequential
        if 'vitals' in patient_sequence:
            vitals = patient_sequence['vitals']
            if vitals.dim() == 3:  # [batch, seq_len, vital_dim]
                vitals = vitals.mean(dim=1)  # [batch, vital_dim]
            features.append(vitals)
        else:
            batch_size = features[0].size(0)
            features.append(torch.zeros(batch_size, self.config.vital_dim, device=self.device))
        
        # Demographics: static features
        if 'demographics' in patient_sequence:
            demo = patient_sequence['demographics']
            features.append(demo)
        else:
            batch_size = features[0].size(0)
            features.append(torch.zeros(batch_size, self.config.demo_dim, device=self.device))
        
        # Concatenate all features
        flattened = torch.cat(features, dim=1)  # [batch, input_dim]
        
        return flattened
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        
        Returns:
            Sampled latent vector [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input data [batch, input_dim]
        
        Returns:
            Latent representation [batch, state_dim]
        """
        # Pass through encoder base
        h = self.encoder_base(x)
        
        if self.variational:
            # VAE: return mean (deterministic encoding for inference)
            mu = self.fc_mu(h)
            return mu
        else:
            # Standard AE
            latent = self.fc_latent(h)
            return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            latent: Latent representation [batch, state_dim]
        
        Returns:
            Reconstructed input [batch, input_dim]
        """
        return self.decoder(latent)
    
    def forward(
        self,
        patient_sequence: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        For training: returns (reconstruction, latent) tuple
        For inference: use encode() method directly
        
        Args:
            patient_sequence: Dictionary of patient data or
                             flattened tensor [batch, input_dim]
            mask: Not used in autoencoder (included for interface consistency)
        
        Returns:
            Tuple of (reconstruction, latent)
            - reconstruction: [batch, input_dim]
            - latent: [batch, state_dim]
        """
        # Handle both dictionary and tensor inputs
        if isinstance(patient_sequence, dict):
            x = self._flatten_patient_data(patient_sequence)
        else:
            x = patient_sequence
        
        # Encode
        h = self.encoder_base(x)
        
        if self.variational:
            # VAE forward pass
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            
            # Reparameterize
            latent = self.reparameterize(mu, logvar)
            
            # Store mu and logvar for loss computation
            self._mu = mu
            self._logvar = logvar
        else:
            # Standard AE
            latent = self.fc_latent(h)
        
        # Decode
        reconstruction = self.decode(latent)
        
        return reconstruction, latent
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute autoencoder loss.
        
        Args:
            x: Original input [batch, input_dim]
            reconstruction: Reconstructed input [batch, input_dim]
            latent: Latent representation [batch, state_dim]
            kl_weight: Weight for KL divergence term (VAE only)
        
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        losses = {
            'reconstruction_loss': recon_loss,
            'total_loss': recon_loss
        }
        
        if self.variational:
            # KL divergence loss for VAE
            # KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * torch.sum(
                1 + self._logvar - self._mu.pow(2) - self._logvar.exp()
            )
            kl_loss = kl_loss / x.size(0)  # Average over batch
            
            losses['kl_loss'] = kl_loss
            losses['total_loss'] = recon_loss + kl_weight * kl_loss
        
        return losses
    
    def get_vae_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get VAE parameters (mu, logvar) from last forward pass.
        
        Returns:
            Tuple of (mu, logvar)
        
        Raises:
            RuntimeError: If not using VAE or if forward hasn't been called
        """
        if not self.variational:
            raise RuntimeError("Not using VAE mode")
        
        if not hasattr(self, '_mu') or not hasattr(self, '_logvar'):
            raise RuntimeError("Must call forward() before getting VAE params")
        
        return self._mu, self._logvar


class DeepAutoencoder(PatientAutoencoder):
    """
    Deeper autoencoder with more layers for complex representations.
    
    Architecture:
        Encoder: input_dim -> 1024 -> 512 -> 256 -> 128 -> latent_dim
        Decoder: latent_dim -> 128 -> 256 -> 512 -> 1024 -> input_dim
    """
    
    def __init__(
        self,
        config: EncoderConfig,
        variational: bool = False
    ):
        """Initialize deep autoencoder."""
        # Don't call super().__init__() to avoid duplicate initialization
        nn.Module.__init__(self)
        BasePatientEncoder.__init__(self, config)
        
        self.variational = variational
        self.input_dim = config.get_total_input_dim()
        
        # Deeper encoder
        encoder_layers = []
        layer_sizes = [self.input_dim, 1024, 512, 256, 128]
        
        for i in range(len(layer_sizes) - 1):
            encoder_layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(layer_sizes[i + 1]),
                nn.Dropout(0.2)
            ])
        
        self.encoder_base = nn.Sequential(*encoder_layers)
        
        # Latent layer
        if variational:
            self.fc_mu = nn.Linear(128, config.state_dim)
            self.fc_logvar = nn.Linear(128, config.state_dim)
        else:
            self.fc_latent = nn.Linear(128, config.state_dim)
        
        # Deeper decoder
        decoder_layers = []
        layer_sizes_dec = [config.state_dim, 128, 256, 512, 1024, self.input_dim]
        
        for i in range(len(layer_sizes_dec) - 1):
            decoder_layers.append(nn.Linear(layer_sizes_dec[i], layer_sizes_dec[i + 1]))
            
            if i < len(layer_sizes_dec) - 2:  # Don't add activation/norm/dropout after last layer
                decoder_layers.extend([
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_sizes_dec[i + 1]),
                    nn.Dropout(0.2)
                ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._initialize_weights()
        
        logger.info(
            f"Initialized Deep{'Variational ' if variational else ''}Autoencoder: "
            f"5-layer architecture, parameters={self.count_parameters():,}"
        )
