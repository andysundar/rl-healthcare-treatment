"""
State Encoder Wrapper
=====================
Thin adapter that sits between flat raw state vectors (e.g. 10-dim synthetic
states) and PatientAutoencoder.  Provides:

  - encode_state        : single state  [raw_dim] -> [state_dim]
  - encode_transitions  : bulk re-encode a list of (s,a,r,s',done) tuples
  - train_on_transitions: fit the autoencoder from scratch on raw state vectors

Usage
-----
from models.encoders import PatientAutoencoder, EncoderConfig
from models.encoders.state_encoder_wrapper import StateEncoderWrapper

enc_cfg = EncoderConfig(lab_dim=10, vital_dim=0, demo_dim=0, state_dim=64)
ae      = PatientAutoencoder(enc_cfg)
wrapper = StateEncoderWrapper(ae, device=torch.device('cpu'), raw_state_dim=10)
wrapper.train_on_transitions(train_data, val_data, epochs=50)
encoded_train = wrapper.encode_transitions(train_data)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Type alias for a single offline RL transition
Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]


class StateEncoderWrapper:
    """
    Wraps a PatientAutoencoder for use with flat state vectors.

    The autoencoder's ``encode(x)`` method accepts a raw tensor of shape
    ``[batch, raw_state_dim]``, so no dict-format conversion is needed when
    the EncoderConfig is set with ``lab_dim=raw_state_dim, vital_dim=0,
    demo_dim=0``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        device: torch.device,
        raw_state_dim: int = 10,
    ) -> None:
        self.encoder = encoder.to(device)
        self.device = device
        self.raw_state_dim = raw_state_dim

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Encode a single raw state vector [raw_dim] -> [state_dim]."""
        self.encoder.eval()
        with torch.no_grad():
            x = torch.tensor(raw_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            z = self.encoder.encode(x)   # [1, state_dim]
        return z.squeeze(0).cpu().numpy()

    def encode_transitions(
        self,
        transitions: List[Transition],
        batch_size: int = 256,
    ) -> List[Transition]:
        """
        Re-encode states in a list of (s, a, r, s', done) tuples.

        Returns a new list where s and s' are replaced by their encoded
        representations.  All other fields (a, r, done) are unchanged.
        """
        if not transitions:
            return []

        states      = np.array([t[0] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)

        enc_states      = self._encode_array(states,      batch_size)
        enc_next_states = self._encode_array(next_states, batch_size)

        return [
            (enc_states[i], transitions[i][1], transitions[i][2],
             enc_next_states[i], transitions[i][4])
            for i in range(len(transitions))
        ]

    def _encode_array(self, arr: np.ndarray, batch_size: int) -> np.ndarray:
        """Encode a 2-D numpy array [N, raw_dim] in batches."""
        self.encoder.eval()
        results = []
        n = len(arr)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.tensor(
                    arr[start:start + batch_size], dtype=torch.float32, device=self.device
                )
                results.append(self.encoder.encode(batch).cpu().numpy())
        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_transitions(
        self,
        train_transitions: List[Transition],
        val_transitions: List[Transition],
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Train the autoencoder using the raw states extracted from transitions.

        Uses MSE reconstruction loss (+ KL term for VAE).  Optimises with
        Adam and applies early stopping on validation loss.

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss'
        """
        train_states = torch.tensor(
            np.array([t[0] for t in train_transitions], dtype=np.float32),
            device=self.device,
        )
        val_states = torch.tensor(
            np.array([t[0] for t in val_transitions], dtype=np.float32),
            device=self.device,
        )

        train_loader = DataLoader(
            TensorDataset(train_states, train_states),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(val_states, val_states),
            batch_size=batch_size, shuffle=False,
        )

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)

        history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
        best_val = float('inf')
        patience, patience_counter = 10, 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            epoch_train_loss = 0.0
            for x_batch, _ in train_loader:
                optimizer.zero_grad()
                reconstruction, latent = self.encoder({"labs": x_batch.unsqueeze(1)})
                loss_dict = self.encoder.compute_loss(x_batch, reconstruction, latent)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * len(x_batch)

            epoch_train_loss /= len(train_states)
            history['train_loss'].append(epoch_train_loss)

            # Validation
            self.encoder.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for x_batch, _ in val_loader:
                    reconstruction, latent = self.encoder({"labs": x_batch.unsqueeze(1)})
                    loss_dict = self.encoder.compute_loss(x_batch, reconstruction, latent)
                    epoch_val_loss += loss_dict['total_loss'].item() * len(x_batch)
            epoch_val_loss /= len(val_states)
            history['val_loss'].append(epoch_val_loss)

            if epoch % max(1, epochs // 5) == 0:
                logger.info(f"  Encoder epoch {epoch}/{epochs}  "
                            f"train={epoch_train_loss:.4f}  val={epoch_val_loss:.4f}")

            # Early stopping
            if epoch_val_loss < best_val - 1e-5:
                best_val = epoch_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_state is not None:
            self.encoder.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Save checkpoint
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / 'encoder_best.pt'
            torch.save(self.encoder.state_dict(), ckpt_path)
            logger.info(f"  Encoder checkpoint saved to {ckpt_path}")

        return history

    def load_checkpoint(self, path: str) -> None:
        """Load encoder weights from a checkpoint file."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(state)
        logger.info(f"Loaded encoder checkpoint from {path}")

    @property
    def state_dim(self) -> int:
        """Output embedding dimensionality."""
        # PatientAutoencoder stores this in its config
        return self.encoder.config.state_dim
