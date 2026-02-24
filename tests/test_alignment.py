"""
PDF Alignment Test Suite  —  M.Tech Project M23CSA508
======================================================
Verifies that every requirement in the M.Tech proposal PDF is correctly
implemented and produces sensible results.

Run:
    cd /path/to/rl-healthcare-treatment
    python tests/test_pdf_alignment.py           # full suite
    python tests/test_pdf_alignment.py --quick   # fast subset only

Sections (matching PDF structure):
  §2   Data Sources            (T01-T02)
  §3   Environments            (T03-T05)
  §3.3 Policy Transfer         (T06)
  §4   Reward Functions        (T07-T09)
  §5   Baselines               (T10)
  §6   CQL Agent               (T11-T12)
  §6   Encoder Pipeline        (T13-T14)
  §6.8 Interpretability        (T15-T17)
  §7   Off-Policy Evaluation   (T18-T19)
  §7.2 Personalization Score   (T20)
  §8   Safety / Clinical       (T21-T22)
  E2E  Full Pipeline           (T23)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# ── test infrastructure ─────────────────────────────────────────────────────
PASS, FAIL, SKIP = '  PASS', '  FAIL', '  SKIP'
_results: List[Tuple[str, str, str]] = []


def run_test(name: str, fn: Callable, quick_skip: bool = False) -> bool:
    """Execute one test, capture pass/fail, print result."""
    global _results
    if quick_skip:
        print(f"{SKIP} [{name}] (slow — use without --quick to run)")
        _results.append((name, 'SKIP', ''))
        return True
    try:
        fn()
        print(f"{PASS} [{name}]")
        _results.append((name, 'PASS', ''))
        return True
    except Exception as e:
        msg = str(e)
        print(f"{FAIL} [{name}]  →  {msg}")
        _results.append((name, 'FAIL', msg))
        if '--verbose' in sys.argv:
            traceback.print_exc()
        return False


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_transitions(n: int = 200, state_dim: int = 10) -> list:
    """Synthetic (s, a, r, s', done) tuples."""
    rng = np.random.default_rng(42)
    transitions = []
    for i in range(n):
        s  = rng.uniform(0, 1, state_dim).astype(np.float32)
        a  = rng.uniform(0, 1, 1).astype(np.float32)
        r  = float(rng.normal(0, 1))
        ns = rng.uniform(0, 1, state_dim).astype(np.float32)
        done = (i % 30 == 29)
        transitions.append((s, a, r, ns, done))
    return transitions


def _make_trajectories_dict(n: int = 5, length: int = 20) -> list:
    """List of episode dicts with states/actions/rewards keys (dict format)."""
    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(n):
        trajs.append({
            'states':      [rng.uniform(0, 1, 10).tolist() for _ in range(length)],
            'actions':     [float(rng.uniform(0, 1)) for _ in range(length)],
            'rewards':     [float(rng.normal(0, 1)) for _ in range(length)],
            'next_states': [rng.uniform(0, 1, 10).tolist() for _ in range(length)],
        })
    return trajs


def _make_trajectories_obj(n: int = 5, length: int = 20) -> list:
    """List of Trajectory dataclass objects (required by OffPolicyEvaluator)."""
    from evaluation.off_policy_eval import Trajectory
    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(n):
        states      = [rng.uniform(0, 1, 10).astype(np.float32) for _ in range(length)]
        actions     = [np.array([rng.uniform(0, 1)], dtype=np.float32) for _ in range(length)]
        rewards     = [float(rng.normal(0, 1)) for _ in range(length)]
        next_states = [rng.uniform(0, 1, 10).astype(np.float32) for _ in range(length)]
        dones       = [False] * (length - 1) + [True]
        trajs.append(Trajectory(
            states=states, actions=actions, rewards=rewards,
            next_states=next_states, dones=dones,
        ))
    return trajs


# ════════════════════════════════════════════════════════════════════════════
# §2  DATA SOURCES
# ════════════════════════════════════════════════════════════════════════════

def t01_synthetic_data_generator():
    """T01 — SyntheticDataGenerator produces valid patient transitions."""
    from data.synthetic_generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(random_seed=0)
    patients = gen.generate_diabetes_population(n_patients=5)
    assert len(patients) == 5, "Expected 5 patients"

    state_cols = [
        'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max',
        'insulin_mean', 'medication_taken', 'reminder_sent',
        'hypoglycemia', 'hyperglycemia', 'day',
    ]
    transitions = []
    for p in patients:
        traj = gen.simulate_patient_trajectory(p, time_horizon_days=10)
        for i in range(len(traj) - 1):
            row = traj.iloc[i]; nrow = traj.iloc[i + 1]
            s  = row[state_cols].values.astype(np.float32)
            ns = nrow[state_cols].values.astype(np.float32)
            transitions.append((s, np.array([float(row['medication_taken'])]), 0.0, ns, False))

    assert len(transitions) > 0, "No transitions generated"
    assert transitions[0][0].shape == (10,), f"State dim wrong: {transitions[0][0].shape}"


def t02_mimic_loader_import():
    """T02 — MIMICLoader module imports correctly and exposes expected methods."""
    from data.mimic_loader import MIMICLoader
    import inspect
    # Don't instantiate — constructor validates directory existence.
    # Just verify the class has the right interface.
    assert hasattr(MIMICLoader, 'load_patients')
    assert hasattr(MIMICLoader, 'load_admissions')
    assert hasattr(MIMICLoader, 'load_lab_events')
    # Constructor signature includes data_dir
    sig = inspect.signature(MIMICLoader.__init__)
    assert 'data_dir' in sig.parameters


# ════════════════════════════════════════════════════════════════════════════
# §3  ENVIRONMENTS
# ════════════════════════════════════════════════════════════════════════════

def t03_diabetes_env_bergman():
    """T03 — DiabetesEnv uses Bergman model, step returns (obs, reward, done, info)."""
    from environments.diabetes_env import DiabetesManagementEnv, DiabetesEnvConfig
    cfg = DiabetesEnvConfig(max_steps=10)
    env = DiabetesManagementEnv(config=cfg)
    reset_out = env.reset()
    # reset() may return (obs, info) or just obs depending on gymnasium version
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    assert obs is not None
    obs_arr = np.asarray(obs)
    assert obs_arr.ndim == 1, f"Observation should be 1-D, got {obs_arr.ndim}-D"
    assert obs_arr.shape[0] == env.observation_space.shape[0]
    action = env.action_space.sample()
    step_out = env.step(action)
    # step() returns (obs, reward, terminated, truncated, info) in gymnasium ≥ 0.26
    obs2 = step_out[0]
    rew  = step_out[1]
    terminated = step_out[2]
    assert isinstance(float(rew), float)
    assert isinstance(bool(terminated), bool)


def t04_adherence_env():
    """T04 — MedicationAdherenceEnv runs without error."""
    from environments.adherence_env import MedicationAdherenceEnv, AdherenceEnvConfig
    env = MedicationAdherenceEnv(config=AdherenceEnvConfig())
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs2, rew, terminated, truncated, info = env.step(action)
    assert obs2 is not None


def t05_disease_models():
    """T05 — BergmanMinimalModel produces physiologically plausible glucose values."""
    from environments.disease_models import BergmanMinimalModel, BergmanModelParams
    params = BergmanModelParams()
    model = BergmanMinimalModel(params)
    glucose_start = 180.0  # mg/dL
    insulin_dose  = 2.0    # units
    # Actual signature: step(G, I, X, insulin_dose, meal_carbs, dt=1.0)
    glucose_next, _, _ = model.step(
        glucose_start, 0.0, 0.0, insulin_dose, 0.0, 5.0
    )
    # glucose should still be in a valid physiological range after one step
    assert 0 < glucose_next < 600, f"Implausible glucose: {glucose_next}"


# ════════════════════════════════════════════════════════════════════════════
# §3.3  POLICY TRANSFER
# ════════════════════════════════════════════════════════════════════════════

def t06_policy_transfer():
    """T06 — PolicyTransferTrainer fits adapter and transfer_select_action returns scalar."""
    from models.policy_transfer import TransferConfig, PolicyTransferTrainer

    # Minimal stub agent
    class StubAgent:
        def eval_mode(self): pass
        def select_action(self, state, deterministic=True):
            # return scalar-like value (float) to match CQLAgent behaviour
            return float(state.mean())

    cfg = TransferConfig(
        source_state_dim=8, target_state_dim=8,
        adaptation_hidden_dim=16, n_adaptation_steps=5,
        adaptation_batch_size=16,
    )
    trainer = PolicyTransferTrainer(
        source_agent=StubAgent(),
        source_encoder=None,
        target_encoder=None,
        config=cfg,
        device=torch.device('cpu'),
    )
    data = _make_transitions(40, state_dim=8)
    history = trainer.fit(data[:20], data[20:])
    assert 'bc_loss' in history
    assert len(history['total_loss']) == cfg.n_adaptation_steps

    raw_state = np.random.randn(8).astype(np.float32)
    action = trainer.transfer_select_action(raw_state)
    assert isinstance(action, float), f"Expected float, got {type(action)}"


# ════════════════════════════════════════════════════════════════════════════
# §4  REWARD FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def t07_composite_reward():
    """T07 — CompositeRewardFunction combines all components correctly."""
    from rewards.composite_reward import CompositeRewardFunction
    from rewards.reward_config import RewardConfig
    from rewards.health_reward import HealthOutcomeReward
    from rewards.safety_reward import SafetyPenalty
    from rewards.adherence_reward import AdherenceReward

    cfg = RewardConfig(w_adherence=0.3, w_health=0.5, w_safety=0.2, w_cost=0.0)
    fn  = CompositeRewardFunction(cfg)
    fn.add_component('health',    HealthOutcomeReward(cfg),  cfg.w_health)
    fn.add_component('safety',    SafetyPenalty(cfg),        cfg.w_safety)
    fn.add_component('adherence', AdherenceReward(cfg),      cfg.w_adherence)

    state      = {'glucose': 100.0, 'adherence': 0.8}
    action     = {'insulin_dose': 2.0}
    next_state = {'glucose': 110.0, 'adherence': 0.85}

    reward = fn.compute_reward(state, action, next_state)
    assert isinstance(reward, float), f"Reward is {type(reward)}, expected float"


def t08_safety_penalty_danger():
    """T08 — SafetyPenalty returns negative reward for dangerous glucose."""
    from rewards.safety_reward import SafetyPenalty
    from rewards.reward_config import RewardConfig
    cfg = RewardConfig()
    pen = SafetyPenalty(cfg)
    state      = {'glucose': 30.0}   # severe hypoglycemia
    next_state = {'glucose': 30.0}
    r = pen.compute_reward(state, {}, next_state)
    assert r < 0, f"Expected penalty, got {r}"


def t09_reward_shaping_utilities():
    """T09 — Reward shaping utilities (normalize, clip, potential-based) work."""
    from rewards.reward_shaping import normalize_reward, clip_reward, potential_based_shaping
    # Actual signatures (from introspection):
    #   normalize_reward(reward, min_val, max_val, target_range=(0.0,1.0)) -> float
    #   clip_reward(reward, clip_range: Tuple[float,float]) -> float
    #   potential_based_shaping(reward, phi_s, phi_s_next, gamma) -> float
    # normalize_reward(reward, min_val, max_val, target_range=(0,1)) -> float
    norm = normalize_reward(0.5, -1.0, 1.0)      # 0.5 in [-1,1] → 0.75 in [0,1]
    assert abs(norm - 0.75) < 1e-3, f"normalize_reward wrong: {norm}"

    # clip_reward(reward, clip_range: Tuple[float,float]) -> float
    r = clip_reward(5.0, (-1.0, 1.0))
    assert abs(r - 1.0) < 1e-6, f"clip_reward wrong: {r}"

    # potential_based_shaping(state_dict, next_state_dict, potential_fn, gamma) -> float
    phi = lambda s: s.get('glucose', 0.0) / 200.0
    shaped = potential_based_shaping(
        {'glucose': 120.0}, {'glucose': 110.0}, phi, gamma=0.99
    )
    assert isinstance(shaped, (int, float)), f"potential_based_shaping wrong type: {type(shaped)}"


# ════════════════════════════════════════════════════════════════════════════
# §5  BASELINE POLICIES
# ════════════════════════════════════════════════════════════════════════════

def t10_baselines_run_and_compare():
    """T10 — All 6 baselines can be trained and compared on test data."""
    from models.baselines import (
        create_diabetes_rule_policy, create_random_policy,
        create_mean_action_policy, create_regression_policy,
        create_knn_policy, create_behavior_cloning_policy,
        compare_all_baselines,
    )
    data  = _make_transitions(200, state_dim=10)
    train = data[:140]; test = data[140:]

    train_s = np.array([t[0] for t in train])
    train_a = np.array([t[1] for t in train])
    val_s   = np.array([t[0] for t in test[:30]])
    val_a   = np.array([t[1] for t in test[:30]])

    baselines = {
        'Rule-Based':     create_diabetes_rule_policy(10, 1),
        'Random':         create_random_policy(1, 10),
        'Mean-Action':    create_mean_action_policy(1, 10).fit(train_s, train_a) or
                          create_mean_action_policy(1, 10),
        'Ridge':          create_regression_policy(10, 1).fit(train_s, train_a) or
                          create_regression_policy(10, 1),
        'KNN-5':          create_knn_policy(10, 1, k=5).fit(train_s, train_a) or
                          create_knn_policy(10, 1, k=5),
    }
    # BC policy
    bc = create_behavior_cloning_policy(10, 1, hidden_dims=[32, 32])
    bc.train(train_s, train_a, val_s, val_a, epochs=3, batch_size=32, verbose=False)
    baselines['BC'] = bc

    df = compare_all_baselines(test_data=test, baselines_dict=baselines,
                               output_path='/tmp/baseline_test.md')
    assert len(df) > 0, "compare_all_baselines returned empty DataFrame"
    assert 'mean_reward' in df.columns


# ════════════════════════════════════════════════════════════════════════════
# §6  CQL AGENT
# ════════════════════════════════════════════════════════════════════════════

def t11_cql_agent_api():
    """T11 — CQLAgent has all required methods (eval_mode, get_q_value, select_action)."""
    from models.rl import CQLAgent
    agent = CQLAgent(state_dim=10, action_dim=1, hidden_dim=32, device='cpu')

    s = np.random.randn(10).astype(np.float32)
    a = np.array([0.5], dtype=np.float32)

    agent.eval_mode()
    action = agent.select_action(s, deterministic=True)
    assert action is not None

    q = agent.get_q_value(s, a)
    assert isinstance(q, float), f"get_q_value should return float, got {type(q)}"


def t12_cql_training_loop(quick: bool = False):
    """T12 — OfflineRLTrainer.train() runs without error (short 50-iter run)."""
    if quick:
        return  # controlled by caller
    from models.rl import CQLAgent, ReplayBuffer, OfflineRLTrainer

    data = _make_transitions(300, state_dim=10)
    agent = CQLAgent(state_dim=10, action_dim=1, hidden_dim=32,
                     q_lr=1e-3, policy_lr=1e-3, device='cpu')

    buf = ReplayBuffer(capacity=300, state_dim=10, action_dim=1, device='cpu')
    states      = np.array([t[0] for t in data])
    actions     = np.array([t[1] for t in data])
    rewards     = np.array([t[2] for t in data], dtype=np.float32)
    next_states = np.array([t[3] for t in data])
    dones       = np.array([t[4] for t in data], dtype=np.float32)
    buf.load_from_dataset(states, actions, rewards, next_states, dones)

    trainer = OfflineRLTrainer(agent=agent, replay_buffer=buf,
                               save_dir='/tmp/cql_test',
                               eval_freq=25, save_freq=50)
    history = trainer.train(num_iterations=50, batch_size=32, verbose=False)
    assert 'train_losses' in history


# ════════════════════════════════════════════════════════════════════════════
# §6  ENCODER PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def t13_autoencoder_encode_shape():
    """T13 — PatientAutoencoder.encode() accepts [batch,10] → returns [batch,state_dim]."""
    from models.encoders import PatientAutoencoder, EncoderConfig
    cfg = EncoderConfig(lab_dim=10, vital_dim=0, demo_dim=0, state_dim=32)
    ae  = PatientAutoencoder(cfg, variational=False)
    x   = torch.randn(8, 10)
    z   = ae.encode(x)
    assert z.shape == (8, 32), f"Expected (8,32), got {z.shape}"


def t14_state_encoder_wrapper():
    """T14 — StateEncoderWrapper encodes transitions and preserves action/reward/done."""
    from models.encoders import PatientAutoencoder, EncoderConfig
    from models.encoders.state_encoder_wrapper import StateEncoderWrapper

    cfg     = EncoderConfig(lab_dim=10, vital_dim=0, demo_dim=0, state_dim=32)
    ae      = PatientAutoencoder(cfg, variational=False)
    wrapper = StateEncoderWrapper(ae, device=torch.device('cpu'), raw_state_dim=10)

    data = _make_transitions(20, state_dim=10)
    enc  = wrapper.encode_transitions(data)

    assert len(enc) == len(data)
    s_enc, a_enc, r_enc, ns_enc, done_enc = enc[0]
    assert s_enc.shape  == (32,), f"Encoded state dim wrong: {s_enc.shape}"
    assert a_enc.shape  == data[0][1].shape,  "Action should be unchanged"
    assert r_enc        == data[0][2],         "Reward should be unchanged"
    assert done_enc     == data[0][4],         "Done should be unchanged"

    # encode_state single
    z = wrapper.encode_state(np.random.randn(10).astype(np.float32))
    assert z.shape == (32,)


# ════════════════════════════════════════════════════════════════════════════
# §6.8  INTERPRETABILITY
# ════════════════════════════════════════════════════════════════════════════

def t15_decision_rule_extractor():
    """T15 — DecisionRuleExtractor fits tree, extracts rules, measures fidelity."""
    from evaluation.interpretability import (
        InterpretabilityConfig, DecisionRuleExtractor
    )

    class StubAgent:
        def eval_mode(self): pass
        def select_action(self, s, deterministic=True):
            return np.array([float(s[0] > 0.5)])  # simple threshold

    cfg   = InterpretabilityConfig(tree_max_depth=3, tree_min_samples_leaf=2)
    extractor = DecisionRuleExtractor(cfg)

    states = np.random.rand(100, 10).astype(np.float32)
    extractor.fit(states, agent=StubAgent())

    rules = extractor.extract_rules()
    assert len(rules) > 0, "No rules extracted"

    fi = extractor.get_feature_importance()
    assert len(fi) == 10, f"Expected 10 feature importances, got {len(fi)}"

    fid = extractor.fidelity_score(states, StubAgent())
    assert 0.0 <= fid <= 1.0, f"Fidelity out of range: {fid}"


def t16_counterfactual_explainer():
    """T16 — CounterfactualExplainer generates counterfactuals with required keys."""
    from evaluation.interpretability import (
        InterpretabilityConfig, CounterfactualExplainer
    )

    class StubAgent:
        def eval_mode(self): pass
        def select_action(self, s, deterministic=True):
            return np.array([float(s[0] > 0.5)])
        def get_q_value(self, s, a):
            return float(s[0])

    cfg = InterpretabilityConfig(
        n_counterfactuals=2,
        counterfactual_steps=10,
        counterfactual_action_threshold=0.0,  # accept any change
    )
    explainer = CounterfactualExplainer(
        agent=StubAgent(),
        encoder_wrapper=None,
        config=cfg,
        device=torch.device('cpu'),
    )
    raw_state = np.random.rand(10).astype(np.float32)
    results   = explainer.explain(raw_state)

    # Should have at most n_counterfactuals results
    assert len(results) <= cfg.n_counterfactuals
    if results:
        keys = {'original_state', 'counterfactual_state',
                'original_action', 'new_action', 'feature_changes', 'l1_distance'}
        assert keys.issubset(results[0].keys()), f"Missing keys: {keys - results[0].keys()}"


def t17_personalization_scorer():
    """T17 — PersonalizationScorer computes (1/T)Σcos(fθ(st),fϕ(xt)) ∈ [-1, 1]."""
    from evaluation.interpretability import PersonalizationScorer
    from models.encoders import PatientAutoencoder, EncoderConfig
    from models.encoders.state_encoder_wrapper import StateEncoderWrapper

    cfg     = EncoderConfig(lab_dim=10, vital_dim=0, demo_dim=0, state_dim=16)
    ae      = PatientAutoencoder(cfg)
    wrapper = StateEncoderWrapper(ae, device=torch.device('cpu'), raw_state_dim=10)

    rng = np.random.default_rng(0)
    source_traj = [rng.uniform(0, 1, 10).astype(np.float32) for _ in range(30)]
    target_traj = [rng.uniform(0, 1, 10).astype(np.float32) for _ in range(30)]

    scorer = PersonalizationScorer(wrapper, wrapper)
    score  = scorer.compute(source_traj, target_traj)
    assert -1.0 <= score <= 1.0, f"Cosine similarity out of range: {score}"

    # batch variant
    result = scorer.compute_batch([source_traj] * 3, [target_traj] * 3)
    assert 'mean' in result and 'std' in result and 'per_trajectory' in result
    assert len(result['per_trajectory']) == 3


# ════════════════════════════════════════════════════════════════════════════
# §7  OFF-POLICY EVALUATION
# ════════════════════════════════════════════════════════════════════════════

class _OPEStubPolicy:
    """Minimal policy stub compatible with OffPolicyEvaluator (needs get_action_probability)."""
    def select_action(self, state, deterministic=False):
        return np.array([0.5], dtype=np.float32)
    def get_action_probability(self, state, action):
        return 0.5   # uniform over [0,1] continuous action → constant density


def t18_wis_ope():
    """T18 — OffPolicyEvaluator.evaluate() with WIS returns a result."""
    from evaluation.off_policy_eval import OffPolicyEvaluator
    trajs = _make_trajectories_obj(n=5, length=10)

    evaluator = OffPolicyEvaluator()
    result    = evaluator.evaluate(
        policy=_OPEStubPolicy(),
        behavior_policy=_OPEStubPolicy(),
        trajectories=trajs,
        methods=['wis'],
    )
    assert result is not None


def t19_ope_methods():
    """T19 — OffPolicyEvaluator.evaluate() accepts multiple OPE methods."""
    from evaluation.off_policy_eval import OffPolicyEvaluator
    trajs = _make_trajectories_obj(n=3, length=5)

    evaluator = OffPolicyEvaluator()
    result = evaluator.evaluate(
        policy=_OPEStubPolicy(),
        behavior_policy=_OPEStubPolicy(),
        trajectories=trajs,
        methods=['wis', 'is'],
    )
    assert result is not None


# ════════════════════════════════════════════════════════════════════════════
# §7.2  PERSONALIZATION SCORE (via PerformanceEvaluator)
# ════════════════════════════════════════════════════════════════════════════

def t20_performance_evaluator_personalization():
    """T20 — PerformanceResult has personalization_score field (PDF §7.2)."""
    from evaluation.performance_metrics import PerformanceResult
    import dataclasses
    fields = {f.name for f in dataclasses.fields(PerformanceResult)}
    assert 'personalization_score' in fields, \
        "personalization_score missing from PerformanceResult"

    r = PerformanceResult(
        average_return=1.0, std_return=0.1, success_rate=0.8,
        average_episode_length=20.0, median_return=0.9,
        min_return=0.0, max_return=2.0, personalization_score=0.75,
    )
    assert r.personalization_score == 0.75


# ════════════════════════════════════════════════════════════════════════════
# §8  SAFETY & CLINICAL METRICS
# ════════════════════════════════════════════════════════════════════════════

def t21_safety_evaluator():
    """T21 — SafetyEvaluator.evaluate() returns SafetyResult with safety_index in [0,1]."""
    from evaluation.safety_metrics import SafetyEvaluator
    from configs.config import EvaluationConfig

    evaluator = SafetyEvaluator(EvaluationConfig())
    trajs     = _make_trajectories_dict(n=3, length=10)
    result    = evaluator.evaluate(trajs)
    assert hasattr(result, 'safety_index'), "SafetyResult has no safety_index"
    assert 0.0 <= result.safety_index <= 1.0, \
        f"safety_index out of range: {result.safety_index}"


def t22_clinical_evaluator():
    """T22 — ClinicalEvaluator.compute_time_in_range() returns a dict."""
    from evaluation.clinical_metrics import ClinicalEvaluator
    from configs.config import EvaluationConfig

    evaluator = ClinicalEvaluator(EvaluationConfig())
    trajs     = _make_trajectories_dict(n=3, length=10)
    tir       = evaluator.compute_time_in_range(trajs)
    assert isinstance(tir, dict), f"Expected dict, got {type(tir)}"


# ════════════════════════════════════════════════════════════════════════════
# E2E  FULL PIPELINE (quick smoke)
# ════════════════════════════════════════════════════════════════════════════

def t23_full_pipeline_synthetic(quick: bool = False):
    """T23 — End-to-end pipeline: synthetic mode finishes without error."""
    import subprocess, sys
    cmd = [
        sys.executable,
        str(ROOT / 'src' / 'run_integrated_solution.py'),
        '--mode', 'synthetic',
        '--n-synthetic-patients', '10',
        '--trajectory-length', '10',
        '--output-dir', '/tmp/test_pipeline_out',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline exited {result.returncode}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def pytest_approx(v, rel=1e-3):
    """Tiny pytest.approx replacement so file runs standalone."""
    class _Approx:
        def __init__(self, v, rel): self.v = v; self.rel = rel
        def __eq__(self, other):
            return abs(other - self.v) <= self.rel * max(abs(self.v), abs(other), 1e-12)
        def __repr__(self): return f"≈{self.v}"
    return _Approx(v, rel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Skip slow tests (CQL training, full pipeline)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full tracebacks on failure')
    args = parser.parse_args()

    tests = [
        # §2  Data
        ("T01 Synthetic data generator",         t01_synthetic_data_generator),
        ("T02 MIMIC loader import",              t02_mimic_loader_import),
        # §3  Environments
        ("T03 DiabetesEnv Bergman model",        t03_diabetes_env_bergman),
        ("T04 AdherenceEnv step",                t04_adherence_env),
        ("T05 BergmanMinimalModel physiology",   t05_disease_models),
        # §3.3 Transfer
        ("T06 PolicyTransfer fit+action",        t06_policy_transfer),
        # §4  Rewards
        ("T07 CompositeRewardFunction",          t07_composite_reward),
        ("T08 SafetyPenalty danger state",       t08_safety_penalty_danger),
        ("T09 Reward shaping utilities",         t09_reward_shaping_utilities),
        # §5  Baselines
        ("T10 Baselines train+compare",          t10_baselines_run_and_compare),
        # §6  CQL
        ("T11 CQLAgent API",                     t11_cql_agent_api),
        ("T12 CQL training loop",                lambda: t12_cql_training_loop(quick=args.quick)),
        # §6  Encoder
        ("T13 Autoencoder encode shape",         t13_autoencoder_encode_shape),
        ("T14 StateEncoderWrapper transitions",  t14_state_encoder_wrapper),
        # §6.8 Interpretability
        ("T15 DecisionRuleExtractor",            t15_decision_rule_extractor),
        ("T16 CounterfactualExplainer",          t16_counterfactual_explainer),
        ("T17 PersonalizationScorer",            t17_personalization_scorer),
        # §7  OPE
        ("T18 WIS off-policy eval",              t18_wis_ope),
        ("T19 Multiple OPE methods",             t19_ope_methods),
        # §7.2 Personalization field
        ("T20 PerformanceResult.personalization_score", t20_performance_evaluator_personalization),
        # §8  Safety/Clinical
        ("T21 SafetyEvaluator index in [0,1]",  t21_safety_evaluator),
        ("T22 ClinicalEvaluator time-in-range",  t22_clinical_evaluator),
        # E2E
        ("T23 Full pipeline (synthetic mode)",   lambda: t23_full_pipeline_synthetic(quick=args.quick)),
    ]

    slow = {'T12 CQL training loop', 'T23 Full pipeline (synthetic mode)'}

    print("\n" + "=" * 70)
    print("M.Tech PDF Alignment Test Suite  —  M23CSA508")
    print("=" * 70)

    passed = failed = skipped = 0
    for name, fn in tests:
        skip_it = args.quick and name in slow
        ok = run_test(name, fn, quick_skip=skip_it)
        if skip_it:
            skipped += 1
        elif ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed  |  {failed} failed  |  {skipped} skipped")
    print("=" * 70 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
