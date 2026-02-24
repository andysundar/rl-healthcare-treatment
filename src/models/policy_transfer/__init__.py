"""
Policy Transfer Module
======================
Domain adaptation: transfer a trained source policy to a new target population.

Classes
-------
TransferConfig          - hyperparameters for the adaptation procedure
PolicyTransferAdapter   - MLP that maps target-domain states to source-domain states
PolicyTransferTrainer   - fits the adapter via behavioural-cloning + cosine alignment
"""

from .transfer import TransferConfig, PolicyTransferAdapter, PolicyTransferTrainer

__all__ = ['TransferConfig', 'PolicyTransferAdapter', 'PolicyTransferTrainer']
