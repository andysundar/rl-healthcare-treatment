#!/usr/bin/env python3
"""
Quick Start Demo - Healthcare RL Environments
==============================================
This script demonstrates basic usage of all environments.
Run this first to verify installation!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from environments import (
    DiabetesManagementEnv,
    MedicationAdherenceEnv,
    PatientSimulator,
    DiseaseSeverity,
    test_environment
)


def demo_diabetes_env():
    """Demonstrate diabetes management environment."""
    print("\n" + "="*70)
    print("DEMO 1: Diabetes Management Environment")
    print("="*70)
    
    # Create environment
    env = DiabetesManagementEnv()
    
    # Run one episode with random policy
    state, info = env.reset(seed=42)
    print(f"\nInitial State:")
    print(f"  Glucose: {state[0]:.1f} mg/dL")
    print(f"  Insulin: {state[1]:.1f} mU/L")
    print(f"  Adherence: {state[5]:.2%}")
    
    total_reward = 0
    for step in range(7):  # One week
        # Simple rule-based action
        glucose = state[0]
        if glucose < 100:
            insulin = 5.0
        elif glucose < 150:
            insulin = 15.0
        else:
            insulin = 30.0
        
        action = np.array([insulin, 0.0, 1.0], dtype=np.float32)
        
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nDay {step+1}: Glucose={state[0]:.1f} mg/dL, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nWeek Total Reward: {total_reward:.2f}")
    print(f"Safety Violations: {info['safety_violations']}")
    
    env.close()


def demo_adherence_env():
    """Demonstrate medication adherence environment."""
    print("\n" + "="*70)
    print("DEMO 2: Medication Adherence Environment")
    print("="*70)
    
    # Create environment
    env = MedicationAdherenceEnv()
    
    # Run one episode
    state, info = env.reset(seed=42)
    print(f"\nInitial Adherence: {state[0]:.2%}")
    
    total_reward = 0
    for day in range(7):  # One week
        # Adaptive policy
        adherence = state[0]
        if adherence < 0.6:
            reminder_type = 2  # Call
            education = 1      # Benefits
        elif adherence < 0.8:
            reminder_type = 1  # SMS
            education = 0      # None
        else:
            reminder_type = 0  # None
            education = 0      # None
        
        action = np.array([reminder_type, education])
        
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Day {day+1}: Adherence={state[0]:.2%}, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nWeek Total Reward: {total_reward:.2f}")
    
    env.close()


def demo_patient_simulator():
    """Demonstrate patient population generation."""
    print("\n" + "="*70)
    print("DEMO 3: Patient Simulator")
    print("="*70)
    
    # Create simulator
    simulator = PatientSimulator(seed=42)
    
    # Generate small cohort
    cohort = simulator.generate_diabetes_cohort(
        n_patients=10,
        severity_distribution={
            DiseaseSeverity.MILD: 0.3,
            DiseaseSeverity.MODERATE: 0.5,
            DiseaseSeverity.SEVERE: 0.2
        }
    )
    
    print(f"\nGenerated {len(cohort)} patients\n")
    
    # Show first 3 patients
    for i, patient in enumerate(cohort[:3]):
        print(f"Patient {i}:")
        print(f"  Age: {patient.demographics.age}, Gender: {patient.demographics.gender}")
        print(f"  BMI: {patient.demographics.bmi:.1f}")
        print(f"  Severity: {patient.clinical.disease_severity.value}")
        print(f"  Initial Glucose: {patient.initial_glucose:.1f} mg/dL")
        print(f"  Baseline Adherence: {patient.clinical.baseline_adherence:.2%}")
        print()
    
    # Get cohort statistics
    stats = simulator.get_cohort_statistics(cohort)
    print("Cohort Statistics:")
    print(f"  Mean Age: {stats['age_mean']:.1f} ± {stats['age_std']:.1f} years")
    print(f"  Mean BMI: {stats['bmi_mean']:.1f} ± {stats['bmi_std']:.1f}")
    print(f"  Mean Adherence: {stats['adherence_mean']:.2%}")


def demo_testing():
    """Demonstrate testing utilities."""
    print("\n" + "="*70)
    print("DEMO 4: Testing Utilities")
    print("="*70)
    
    env = DiabetesManagementEnv()
    
    print("\nRunning basic environment tests...")
    results = test_environment(env, n_episodes=5, verbose=False)
    
    print(f"\nTest Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
    if 'mean_safety_violations' in results:
        print(f"  Mean Safety Violations: {results['mean_safety_violations']:.2f}")
    
    env.close()


def demo_generate_offline_dataset():
    """Demonstrate generating offline dataset for CQL."""
    print("\n" + "="*70)
    print("DEMO 5: Generate Offline Dataset for CQL Training")
    print("="*70)
    
    # Simple behavior policy
    def behavior_policy(state):
        glucose = state[0]
        noise = np.random.normal(0, 3)
        
        if glucose < 100:
            insulin = max(0, 5 + noise)
        elif glucose < 150:
            insulin = max(0, 15 + noise)
        else:
            insulin = max(0, 30 + noise)
        
        return np.array([insulin, 0.0, 1.0], dtype=np.float32)
    
    # Generate dataset
    dataset = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }
    
    print("\nGenerating offline dataset with 5 patients, 7 days each...")
    
    for patient_id in range(5):
        env = DiabetesManagementEnv(patient_id=patient_id)
        state, _ = env.reset(seed=42)
        
        for day in range(7):
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            dataset['states'].append(state)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['next_states'].append(next_state)
            dataset['dones'].append(terminated or truncated)
            
            if terminated or truncated:
                break
            
            state = next_state
        
        env.close()
    
    # Convert to arrays
    for key in dataset:
        dataset[key] = np.array(dataset[key])
    
    print(f"\nDataset Generated:")
    print(f"  Total Transitions: {len(dataset['states'])}")
    print(f"  States Shape: {dataset['states'].shape}")
    print(f"  Actions Shape: {dataset['actions'].shape}")
    print(f"  Mean Reward: {np.mean(dataset['rewards']):.2f}")
    print(f"  Episodes Completed: {sum(dataset['dones'])}")
    
    print("\n✓ This dataset can now be used to train your CQL agent!")
    
    return dataset


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("HEALTHCARE RL ENVIRONMENTS - QUICK START DEMO")
    print("="*70)
    
    try:
        demo_diabetes_env()
        demo_adherence_env()
        demo_patient_simulator()
        demo_testing()
        dataset = demo_generate_offline_dataset()
        
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext Steps:")
        print("1. Explore the examples/ directory for more detailed examples")
        print("2. Read INTEGRATION_GUIDE.md for thesis integration")
        print("3. Use the generated dataset to train your CQL agent")
        print("4. Test your policy with PolicyEvaluator")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed the requirements:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
