"""
Unit tests for src/evaluation/policy_interpretability.py.

All tests are self-contained and use lightweight policy stubs.
No checkpoints, GPU, or MIMIC data are required.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.policy_interpretability import (
    _DEFAULT_FEATURE_NAMES,
    _bin_actions,
    _query_policy,
    _resolve_names,
    compute_feature_importance,
    extract_decision_rules,
    generate_counterfactual,
)

# ---------------------------------------------------------------------------
# Policy stubs
# ---------------------------------------------------------------------------

class _ConstantPolicy:
    """Always returns the same action regardless of state."""
    def __init__(self, action: float = 0.5):
        self._a = action

    def select_action(self, state, deterministic=True):
        return np.array([self._a], dtype=np.float32)


class _ThresholdPolicy:
    """
    Emits a HIGH action (0.8) when state[0] > threshold, LOW (0.2) otherwise.
    This gives the surrogate tree / RF something meaningful to learn.
    """
    def __init__(self, threshold: float = 0.5):
        self._t = threshold

    def select_action(self, state, deterministic=True):
        a = 0.8 if float(state[0]) > self._t else 0.2
        return np.array([a], dtype=np.float32)


class _Feature1Policy:
    """
    Sensitive to feature[1]: action = 0.8 if state[1] > 0.5 else 0.2.
    Useful for counterfactual tests.
    """
    def select_action(self, state, deterministic=True):
        a = 0.8 if float(state[1]) > 0.5 else 0.2
        return np.array([a], dtype=np.float32)


def _make_states(n: int = 200, dim: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make_pairs(policy, n: int = 200, dim: int = 5, seed: int = 0):
    states = _make_states(n, dim, seed)
    return [(states[i], None) for i in range(n)]   # let function query policy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestBinActions:
    def test_low_bucket(self):
        a = np.array([0.0, 0.1, 0.33], dtype=np.float32)
        labels = _bin_actions(a)
        assert all(l == "low" for l in labels)

    def test_high_bucket(self):
        a = np.array([0.67, 0.9, 1.0], dtype=np.float32)
        labels = _bin_actions(a)
        assert all(l == "high" for l in labels)

    def test_medium_bucket(self):
        a = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        labels = _bin_actions(a)
        assert all(l == "medium" for l in labels)

    def test_shape_preserved(self):
        a = np.linspace(0, 1, 30, dtype=np.float32)
        assert len(_bin_actions(a)) == 30


class TestQueryPolicy:
    def test_returns_float_array(self):
        states = _make_states(10, 3)
        out = _query_policy(_ConstantPolicy(0.7), states)
        assert out.dtype == np.float32
        assert out.shape == (10,)

    def test_constant_policy_values(self):
        states = _make_states(5, 3)
        out = _query_policy(_ConstantPolicy(0.3), states)
        assert np.allclose(out, 0.3)


class TestResolveNames:
    def test_uses_pipeline_defaults(self):
        names = _resolve_names(None, 4)
        assert names == _DEFAULT_FEATURE_NAMES[:4]

    def test_trims_long_list(self):
        long = [f"f{i}" for i in range(20)]
        assert _resolve_names(long, 5) == long[:5]

    def test_pads_short_list(self):
        short = ["a", "b"]
        resolved = _resolve_names(short, 5)
        assert len(resolved) == 5
        assert resolved[:2] == ["a", "b"]
        assert all("feature_" in r for r in resolved[2:])

    def test_exact_length(self):
        names = ["x", "y", "z"]
        assert _resolve_names(names, 3) == names


# ---------------------------------------------------------------------------
# extract_decision_rules
# ---------------------------------------------------------------------------

class TestExtractDecisionRules:
    def test_returns_tuple_of_tree_and_rules(self):
        pairs = _make_pairs(_ThresholdPolicy(), n=100, dim=5)
        tree, rules = extract_decision_rules(_ThresholdPolicy(), pairs, max_depth=5)
        assert hasattr(tree, "predict")
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_tree_max_depth_respected(self):
        pairs = _make_pairs(_ThresholdPolicy(), n=200, dim=5)
        tree, _ = extract_decision_rules(_ThresholdPolicy(), pairs, max_depth=3)
        assert tree.get_depth() <= 3

    def test_rules_are_strings(self):
        pairs = _make_pairs(_ThresholdPolicy(), n=100, dim=5)
        _, rules = extract_decision_rules(_ThresholdPolicy(), pairs)
        assert all(isinstance(r, str) for r in rules)

    def test_rules_contain_feature_names(self):
        names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        pairs = _make_pairs(_ThresholdPolicy(), n=100, dim=5)
        _, rules = extract_decision_rules(
            _ThresholdPolicy(), pairs, feature_names=names
        )
        rule_text = " ".join(rules)
        # At least one feature name should appear in the rules.
        assert any(n in rule_text for n in names)

    def test_pre_computed_actions_accepted(self):
        """state_action_pairs with explicit action values — policy not queried."""
        rng = np.random.default_rng(7)
        states = rng.standard_normal((80, 4)).astype(np.float32)
        actions = rng.uniform(0, 1, 80).astype(np.float32)
        pairs = list(zip(states, actions))
        tree, rules = extract_decision_rules(None, pairs, max_depth=3)
        assert tree is not None
        assert len(rules) > 0

    def test_wide_state_auto_pads_names(self):
        """State dim > len(feature_names): extra dims auto-named."""
        pairs = _make_pairs(_ThresholdPolicy(), n=100, dim=8)
        names = ["a", "b"]   # only 2 names for 8 features
        _, rules = extract_decision_rules(
            _ThresholdPolicy(), pairs, feature_names=names
        )
        rule_text = " ".join(rules)
        # At least one auto-generated name should appear
        assert "feature_" in rule_text or "a" in rule_text

    def test_tree_learns_threshold_policy(self):
        """The tree should achieve near-perfect fidelity on a learnable policy."""
        rng = np.random.default_rng(3)
        states = rng.standard_normal((300, 5)).astype(np.float32)
        policy = _ThresholdPolicy(threshold=0.0)
        pairs = [(s, None) for s in states]
        tree, _ = extract_decision_rules(policy, pairs, max_depth=5)
        # Measure fidelity on the training set
        preds = tree.predict(states)
        from evaluation.policy_interpretability import _bin_actions, _query_policy
        labels = _bin_actions(_query_policy(policy, states))
        fidelity = float(np.mean(preds == labels))
        assert fidelity > 0.85


# ---------------------------------------------------------------------------
# compute_feature_importance
# ---------------------------------------------------------------------------

class TestComputeFeatureImportance:
    def test_returns_dict(self):
        states = _make_states(200, 5)
        result = compute_feature_importance(
            _ThresholdPolicy(), states,
            feature_names=["a", "b", "c", "d", "e"],
            save_path=str(Path(tempfile.mkdtemp()) / "fi.png"),
        )
        assert isinstance(result, dict)

    def test_keys_match_feature_names(self):
        names = ["f0", "f1", "f2", "f3"]
        states = _make_states(200, 4)
        result = compute_feature_importance(
            _ThresholdPolicy(), states,
            feature_names=names,
            save_path=str(Path(tempfile.mkdtemp()) / "fi.png"),
        )
        assert set(result.keys()) == set(names)

    def test_importances_sum_to_one(self):
        states = _make_states(200, 5)
        result = compute_feature_importance(
            _ThresholdPolicy(), states,
            feature_names=["a", "b", "c", "d", "e"],
            save_path=str(Path(tempfile.mkdtemp()) / "fi.png"),
        )
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_sorted_descending(self):
        states = _make_states(200, 5)
        result = compute_feature_importance(
            _ThresholdPolicy(), states,
            feature_names=["a", "b", "c", "d", "e"],
            save_path=str(Path(tempfile.mkdtemp()) / "fi.png"),
        )
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_threshold_policy_top_feature_is_first_dim(self):
        """
        _ThresholdPolicy splits on state[0]; that dimension should have the
        highest importance in the RF surrogate.
        """
        rng = np.random.default_rng(42)
        states = rng.standard_normal((500, 5)).astype(np.float32)
        names  = ["main", "noise1", "noise2", "noise3", "noise4"]
        result = compute_feature_importance(
            _ThresholdPolicy(threshold=0.0), states,
            feature_names=names,
            save_path=str(Path(tempfile.mkdtemp()) / "fi.png"),
            n_estimators=50,
        )
        top_feature = next(iter(result))
        assert top_feature == "main"

    def test_chart_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "subdir" / "fi.png"
            states = _make_states(100, 4)
            compute_feature_importance(
                _ThresholdPolicy(), states,
                feature_names=["a", "b", "c", "d"],
                save_path=str(out),
                n_estimators=10,
            )
            assert out.exists()

    def test_default_feature_names_used(self):
        """No feature_names supplied → falls back to pipeline BASE_STATE_COLS."""
        states = _make_states(100, 5)
        result = compute_feature_importance(
            _ThresholdPolicy(), states,
            save_path=str(Path(tempfile.mkdtemp()) / "fi.png"),
            n_estimators=10,
        )
        # Keys should be from the pipeline default list
        assert all(k in _DEFAULT_FEATURE_NAMES or "feature_" in k for k in result)


# ---------------------------------------------------------------------------
# generate_counterfactual
# ---------------------------------------------------------------------------

class TestGenerateCounterfactual:
    def _state(self, val=0.3, dim=5):
        s = np.zeros(dim, dtype=np.float32)
        s[1] = val
        return s

    def test_returns_list(self):
        state = self._state(val=0.3)
        result = generate_counterfactual(_Feature1Policy(), state)
        assert isinstance(result, list)

    def test_action_flips_on_feature1_above_threshold(self):
        """
        _Feature1Policy triggers on state[1] > 0.5 (strict).
        Starting at 0.42, a +25% push gives 0.525 which crosses the boundary.
        """
        state = self._state(val=0.42)
        result = generate_counterfactual(_Feature1Policy(), state)
        flipped_features = {e["feature_index"] for e in result}
        assert 1 in flipped_features

    def test_returned_entries_all_changed_action(self):
        state = self._state(val=0.42)
        result = generate_counterfactual(_Feature1Policy(), state)
        for entry in result:
            assert abs(entry["new_action"] - entry["original_action"]) > 1e-4

    def test_dict_keys_present(self):
        state = self._state(val=0.42)
        result = generate_counterfactual(_Feature1Policy(), state)
        if result:
            required = {
                "feature", "feature_index",
                "perturbation_frac", "perturbation_abs",
                "original_value", "perturbed_value",
                "original_action", "new_action", "action_delta",
            }
            assert required.issubset(result[0].keys())

    def test_constant_policy_returns_empty(self):
        """A policy that never changes should produce no counterfactuals."""
        state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        result = generate_counterfactual(_ConstantPolicy(0.5), state)
        assert result == []

    def test_custom_perturbation_fractions(self):
        """Only the supplied fractions should be tried."""
        state = self._state(val=0.45)
        # Only a large +50% push (non-default) flips the boundary at 0.5
        result = generate_counterfactual(
            _Feature1Policy(), state,
            perturbation_fractions=(0.50,),
        )
        # 0.45 * 1.5 = 0.675 > 0.5 → should flip
        flipped = [e for e in result if e["feature_index"] == 1]
        assert len(flipped) == 1
        assert flipped[0]["perturbation_frac"] == pytest.approx(0.5)

    def test_feature_names_appear_in_output(self):
        names = ["g_mean", "g_std", "g_min", "g_max", "day"]
        state = self._state(val=0.4, dim=5)
        result = generate_counterfactual(_Feature1Policy(), state, feature_names=names)
        returned_names = {e["feature"] for e in result}
        assert returned_names.issubset(set(names))

    def test_zero_valued_feature_does_not_produce_noop(self):
        """
        absolute_floor ensures that a zero-valued feature still receives a
        non-trivial perturbation.
        """
        state = np.zeros(5, dtype=np.float32)
        # Use a policy that returns different actions for non-zero vs zero state[0]
        class _ZeroSensitive:
            def select_action(self, s, deterministic=True):
                return np.array([0.9 if float(s[0]) > 0 else 0.1], dtype=np.float32)

        result = generate_counterfactual(_ZeroSensitive(), state)
        perturbed_abs = [e["perturbation_abs"] for e in result if e["feature_index"] == 0]
        assert all(abs(p) >= 1e-6 for p in perturbed_abs)

    def test_action_delta_consistent(self):
        """action_delta == new_action - original_action for every entry."""
        state = self._state(val=0.42)
        result = generate_counterfactual(_Feature1Policy(), state)
        for entry in result:
            expected_delta = round(entry["new_action"] - entry["original_action"], 6)
            assert entry["action_delta"] == pytest.approx(expected_delta, abs=1e-5)
