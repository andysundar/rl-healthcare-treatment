import numpy as np

from src.models.safety.config import SafetyConfig
from src.models.safety.safety_layer import SafetyLayer


def test_discrete_safety_mask_blocks_obviously_unsafe_actions():
    layer = SafetyLayer(SafetyConfig())
    low_glucose_state = np.array([65.0], dtype=np.float32)
    # for 4 buckets, high buckets (2,3) should be masked under low glucose
    masked = layer.apply_discrete_action_mask(low_glucose_state, action_idx=3, n_actions=4)
    assert masked in [0, 1]

    high_glucose_state = np.array([260.0], dtype=np.float32)
    masked2 = layer.apply_discrete_action_mask(high_glucose_state, action_idx=0, n_actions=4)
    assert masked2 != 0
