# Integration Guide: Healthcare RL Evaluation Framework

## For Andy's M.Tech Project

This guide shows you exactly how to integrate the evaluation framework into your existing healthcare RL project.

## Quick Integration Steps

### 1. Copy Framework to Your Project

```bash
# From healthcare_rl_evaluation directory
cp -r src/evaluation /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl/src/
cd /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl
pip install scipy matplotlib seaborn pandas pyyaml
```

### 2. Update Your Imports

In your training script:

```python
# Add at top of file
from src.evaluation import (
    EvaluationConfig,
    OffPolicyEvaluator,
    SafetyEvaluator,
    ClinicalEvaluator,
    PerformanceEvaluator,
    Trajectory
)
```

### 3. Integrate with CQL Training

After your CQL training loop:

```python
# After training your CQL agent
def evaluate_policy(cql_agent, test_data):
    # Initialize configuration
    config = EvaluationConfig()
    
    # 1. SAFETY EVALUATION
    print("Running safety evaluation...")
    safety_evaluator = SafetyEvaluator(config)
    
    # Convert your trajectories to evaluation format
    eval_trajectories = convert_to_eval_format(test_data)
    
    safety_result = safety_evaluator.evaluate(eval_trajectories)
    
    # 2. CLINICAL EVALUATION
    print("Running clinical evaluation...")
    clinical_evaluator = ClinicalEvaluator(config)
    
    # You need baseline for comparison
    baseline_trajectories = load_baseline_trajectories()  # or generate
    
    clinical_result = clinical_evaluator.evaluate(
        eval_trajectories,
        baseline_trajectories
    )
    
    # 3. OFF-POLICY EVALUATION
    print("Running OPE...")
    
    # Wrap your Q-network
    def q_function(state, action):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        with torch.no_grad():
            return cql_agent.critic(state_tensor, action_tensor).item()
    
    ope_evaluator = OffPolicyEvaluator(
        gamma=0.99,
        q_function=q_function
    )
    
    # Convert to Trajectory objects for OPE
    ope_trajectories = []
    for traj in eval_trajectories[:100]:  # Use subset
        states = np.array([list(s.values()) for s in traj['states']])
        actions = np.array([list(a.values()) for a in traj['actions']])
        rewards = np.array(traj['rewards'])
        next_states = np.array([list(s.values()) for s in traj['next_states']])
        dones = np.array(traj['dones'], dtype=float)
        
        traj_obj = Trajectory(states, actions, rewards, next_states, dones)
        ope_trajectories.append(traj_obj)
    
    # Wrap your policy
    class PolicyWrapper:
        def __init__(self, cql_agent):
            self.cql_agent = cql_agent
        
        def __call__(self, state, deterministic=False):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = self.cql_agent.actor(state_tensor)
            return action.numpy()
        
        def get_action_probability(self, state, action):
            # For Gaussian policy
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            with torch.no_grad():
                mean = self.cql_agent.actor(state_tensor)
                log_prob = self.cql_agent.actor.log_prob(mean, action_tensor)
            return torch.exp(log_prob).item()
    
    policy = PolicyWrapper(cql_agent)
    behavior_policy = load_behavior_policy()  # or use same as policy
    
    ope_results = ope_evaluator.evaluate(
        policy,
        behavior_policy,
        ope_trajectories,
        methods=['wis', 'dr']
    )
    
    ope_evaluator.compare_methods(ope_results)
    
    return {
        'safety': safety_result,
        'clinical': clinical_result,
        'ope': ope_results
    }

def convert_to_eval_format(test_data):
    """Convert your MIMIC-III data to evaluation format."""
    trajectories = []
    
    for episode in test_data:
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        for step in episode:
            # Extract from your data format
            state = {
                'glucose': step['glucose_value'],
                'bp_systolic': step['bp_systolic_value'],
                'adherence': step['adherence_score'],
                # Add other state variables
            }
            
            action = {
                'medication_dosage': step['action_dosage'],
                'reminder_frequency': step['action_reminder']
                # Add other action components
            }
            
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(step['reward'])
            trajectory['next_states'].append(step['next_state'])
            trajectory['dones'].append(step['done'])
        
        trajectories.append(trajectory)
    
    return trajectories
```

### 4. Add to Your Main Training Script

```python
# In main.py or train_cql.py

if __name__ == "__main__":
    # Your existing training code
    # ...
    
    # After training
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    evaluation_results = evaluate_policy(cql_agent, test_data)
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'safety_index': evaluation_results['safety'].safety_index,
            'violation_rate': evaluation_results['safety'].violation_rate,
            'adverse_event_rate': evaluation_results['clinical'].adverse_event_rate,
            'wis_value': evaluation_results['ope']['wis'].value_estimate if 'wis' in evaluation_results['ope'] else None,
        }, f, indent=2)
```

### 5. Using with MIMIC-III Data

```python
# Example: Load MIMIC-III and evaluate
from src.data.mimic_processor import MIMICProcessor  # Your existing module
from src.evaluation import SafetyEvaluator, EvaluationConfig

# Load your processed MIMIC data
mimic_processor = MIMICProcessor(data_dir="/path/to/MIMIC-III")
test_episodes = mimic_processor.load_test_data()

# Convert to evaluation format
eval_trajectories = []
for episode in test_episodes:
    traj = {
        'states': episode.states,  # Your state format
        'actions': episode.actions,
        'rewards': episode.rewards,
        'next_states': episode.next_states,
        'dones': episode.dones
    }
    eval_trajectories.append(traj)

# Evaluate
config = EvaluationConfig()
safety_evaluator = SafetyEvaluator(config)
safety_result = safety_evaluator.evaluate(eval_trajectories)

print(f"Safety Index: {safety_result.safety_index:.3f}")
print(f"Violation Rate: {safety_result.violation_rate:.3f}")
```

## Thesis Integration Checklist

For your thesis, you need to demonstrate:

- [x] **OPE Methods**: All four methods (IS, WIS, DR, DM) implemented
- [x] **Safety Metrics**: Safety index, violation rates, severity
- [x] **Clinical Metrics**: Health improvements, time in range, adverse events
- [x] **Baseline Comparison**: Compare CQL vs standard care
- [x] **Statistical Significance**: Built-in confidence intervals

## Results Formatting for Thesis

```python
# Generate thesis-ready results table
from src.evaluation import PolicyComparator
import pandas as pd

# Compare your CQL policy with baselines
policies = {
    'CQL': cql_agent,
    'Behavioral_Cloning': bc_agent,
    'Standard_Care': standard_care_policy
}

comparator = PolicyComparator(config)
comparison_df = comparator.compare_policies(
    policies,
    test_data,
    evaluators=[
        lambda t: SafetyEvaluator(config).evaluate(t),
        lambda t: ClinicalEvaluator(config).evaluate(t, baseline_trajectories)
    ]
)

# Save for thesis
comparison_df.to_latex('results/policy_comparison.tex')
comparison_df.to_csv('results/policy_comparison.csv')
```

## Visualization for Thesis

```python
from src.evaluation import EvaluationVisualizer

visualizer = EvaluationVisualizer(config)

# Figure 1: Policy Comparison
visualizer.plot_comparison(
    comparison_df,
    metrics=['avg_return', 'safety_index'],
    save_path='thesis/figures/policy_comparison.pdf'
)

# Figure 2: Safety Analysis
visualizer.plot_safety_violations(
    eval_trajectories,
    variables=['glucose', 'bp_systolic'],
    save_path='thesis/figures/safety_violations.pdf'
)

# Figure 3: Clinical Outcomes
visualizer.plot_health_metrics(
    eval_trajectories,
    metrics=['glucose', 'bp_systolic', 'hba1c'],
    save_path='thesis/figures/health_metrics.pdf'
)
```

## Next Steps

1. Copy framework to your project
2. Test with synthetic data first (use provided example)
3. Integrate with your MIMIC-III pipeline
4. Run full evaluation on trained CQL agent
5. Generate results for thesis

## Troubleshooting

**Q: What if my state/action format is different?**
A: Just write a converter function like `convert_to_eval_format()` above

**Q: Can I use this with my AgeBucketing?**
A: Yes! The evaluator works with any dictionary state representation

**Q: How do I handle missing metrics?**
A: The evaluators skip missing metrics gracefully - just include what you have

**Q: What if I don't have a baseline?**
A: You can create a simple baseline (random policy, behavioral cloning, or rule-based)

## Contact for Issues

If you encounter any integration issues, check the examples directory or reach out.

Good luck with your thesis completion!
