"""
Example: Patient Simulator
===========================
Demonstrates usage of the patient simulator for generating diverse populations.
"""

import numpy as np
import sys
sys.path.append('..')

from environments import (
    PatientSimulator,
    DiseaseSeverity,
    DiabetesManagementEnv,
    MedicationAdherenceEnv
)


def generate_diabetes_cohort_example():
    """Generate a diverse diabetes patient cohort."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Generating Diabetes Cohort")
    print("="*70)
    
    simulator = PatientSimulator(seed=42)
    
    # Generate 100 patients
    cohort = simulator.generate_diabetes_cohort(
        n_patients=100,
        severity_distribution={
            DiseaseSeverity.MILD: 0.3,
            DiseaseSeverity.MODERATE: 0.5,
            DiseaseSeverity.SEVERE: 0.2
        }
    )
    
    print(f"\nGenerated {len(cohort)} patients")
    
    # Show first 5 patients
    print("\nSample Patients:")
    print("-"*70)
    
    for i, patient in enumerate(cohort[:5]):
        print(f"\nPatient {i}:")
        print(f"  Demographics:")
        print(f"    Age: {patient.demographics.age}")
        print(f"    Gender: {patient.demographics.gender}")
        print(f"    BMI: {patient.demographics.bmi:.1f}")
        print(f"    Ethnicity: {patient.demographics.ethnicity}")
        print(f"    SES: {patient.demographics.socioeconomic_status}")
        print(f"  Clinical:")
        print(f"    Severity: {patient.clinical.disease_severity.value}")
        print(f"    Comorbidities: {', '.join(patient.clinical.comorbidities) if patient.clinical.comorbidities else 'None'}")
        print(f"    Baseline Adherence: {patient.clinical.baseline_adherence:.2%}")
        print(f"  Physiological:")
        print(f"    Initial Glucose: {patient.initial_glucose:.1f} mg/dL")
        print(f"    Initial Insulin: {patient.initial_insulin:.1f} mU/L")
    
    # Get cohort statistics
    stats = simulator.get_cohort_statistics(cohort)
    
    print("\n" + "="*70)
    print("COHORT STATISTICS")
    print("="*70)
    print(f"Total Patients: {stats['n_patients']}")
    print(f"\nDemographics:")
    print(f"  Age: {stats['age_mean']:.1f} ± {stats['age_std']:.1f} years")
    print(f"  BMI: {stats['bmi_mean']:.1f} ± {stats['bmi_std']:.1f}")
    print(f"  Gender Distribution:")
    for gender, proportion in stats['gender_distribution'].items():
        print(f"    {gender}: {proportion:.1%}")
    
    print(f"\nClinical:")
    print(f"  Baseline Adherence: {stats['adherence_mean']:.2%} ± {stats['adherence_std']:.2%}")
    print(f"  Severity Distribution:")
    for severity, proportion in stats['severity_distribution'].items():
        print(f"    {severity}: {proportion:.1%}")


def generate_adherence_cohort_example():
    """Generate a cohort focused on adherence."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Generating Adherence-Focused Cohort")
    print("="*70)
    
    simulator = PatientSimulator(seed=42)
    
    # Generate cohort with varied baseline adherence
    cohort = simulator.generate_adherence_cohort(
        n_patients=50,
        baseline_adherence_distribution={
            'mean': 0.65,
            'std': 0.20,
            'min': 0.2,
            'max': 0.95
        }
    )
    
    print(f"\nGenerated {len(cohort)} patients for adherence study")
    
    # Analyze adherence distribution
    adherence_levels = [p.initial_adherence for p in cohort]
    
    print("\nAdherence Distribution:")
    print(f"  Mean: {np.mean(adherence_levels):.2%}")
    print(f"  Std: {np.std(adherence_levels):.2%}")
    print(f"  Min: {np.min(adherence_levels):.2%}")
    print(f"  Max: {np.max(adherence_levels):.2%}")
    
    # Categorize patients
    low_adherence = sum(1 for a in adherence_levels if a < 0.5)
    medium_adherence = sum(1 for a in adherence_levels if 0.5 <= a < 0.8)
    high_adherence = sum(1 for a in adherence_levels if a >= 0.8)
    
    print("\nPatient Categories:")
    print(f"  Low Adherence (<50%): {low_adherence} ({low_adherence/len(cohort):.1%})")
    print(f"  Medium Adherence (50-80%): {medium_adherence} ({medium_adherence/len(cohort):.1%})")
    print(f"  High Adherence (>80%): {high_adherence} ({high_adherence/len(cohort):.1%})")


def test_patient_variability():
    """Test physiological variability across patients."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Patient Physiological Variability")
    print("="*70)
    
    simulator = PatientSimulator(seed=42)
    
    # Generate patients with different severities
    severities = [DiseaseSeverity.MILD, DiseaseSeverity.MODERATE, DiseaseSeverity.SEVERE]
    
    for severity in severities:
        print(f"\n--- {severity.value.upper()} Patients ---")
        
        cohort = simulator.generate_diabetes_cohort(
            n_patients=30,
            severity_distribution={severity: 1.0}
        )
        
        # Analyze Bergman parameters
        p1_values = [p.bergman_params.p1 for p in cohort]
        p3_values = [p.bergman_params.p3 for p in cohort]
        glucose_values = [p.initial_glucose for p in cohort]
        
        print(f"Glucose Effectiveness (p1):")
        print(f"  Mean: {np.mean(p1_values):.6f}")
        print(f"  Std: {np.std(p1_values):.6f}")
        
        print(f"Insulin Sensitivity (p3):")
        print(f"  Mean: {np.mean(p3_values):.8f}")
        print(f"  Std: {np.std(p3_values):.8f}")
        
        print(f"Initial Glucose:")
        print(f"  Mean: {np.mean(glucose_values):.1f} mg/dL")
        print(f"  Std: {np.std(glucose_values):.1f} mg/dL")


def simulate_population_outcomes():
    """Simulate outcomes across a population with a simple policy."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Population-Level Outcomes")
    print("="*70)
    
    simulator = PatientSimulator(seed=42)
    
    # Generate diverse cohort
    cohort = simulator.generate_diabetes_cohort(n_patients=20)
    
    # Simple policy
    def simple_policy(state):
        glucose = state[0]
        if glucose < 80:
            insulin = 5
        elif glucose < 130:
            insulin = 15
        elif glucose < 180:
            insulin = 25
        else:
            insulin = 40
        return np.array([insulin, 0.0, 1.0], dtype=np.float32)
    
    print("\nSimulating 20 patients over 30 days...")
    
    outcomes = []
    
    for i, patient in enumerate(cohort):
        env = DiabetesManagementEnv(patient_id=patient.demographics.patient_id)
        
        state, _ = env.reset(seed=42)
        total_reward = 0
        glucose_levels = []
        
        for day in range(30):
            action = simple_policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            glucose_levels.append(state[0])
            
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        
        outcomes.append({
            'patient_id': patient.demographics.patient_id,
            'severity': patient.clinical.disease_severity.value,
            'total_reward': total_reward,
            'mean_glucose': np.mean(glucose_levels),
            'glucose_variability': np.std(glucose_levels),
            'safety_violations': metrics.get('safety_violations', 0),
            'time_in_range': sum(80 <= g <= 130 for g in glucose_levels) / len(glucose_levels)
        })
        
        env.close()
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/20 patients...")
    
    print("\n" + "="*70)
    print("POPULATION OUTCOMES SUMMARY")
    print("="*70)
    
    # Overall statistics
    print(f"\nOverall Performance:")
    print(f"  Mean Reward: {np.mean([o['total_reward'] for o in outcomes]):.2f}")
    print(f"  Mean Glucose: {np.mean([o['mean_glucose'] for o in outcomes]):.1f} mg/dL")
    print(f"  Time in Range: {np.mean([o['time_in_range'] for o in outcomes]):.1%}")
    print(f"  Total Safety Violations: {sum([o['safety_violations'] for o in outcomes])}")
    
    # By severity
    print(f"\nBy Severity:")
    for severity in ['mild', 'moderate', 'severe']:
        severity_outcomes = [o for o in outcomes if o['severity'] == severity]
        if severity_outcomes:
            print(f"  {severity.upper()}:")
            print(f"    N: {len(severity_outcomes)}")
            print(f"    Mean Reward: {np.mean([o['total_reward'] for o in severity_outcomes]):.2f}")
            print(f"    Time in Range: {np.mean([o['time_in_range'] for o in severity_outcomes]):.1%}")
            print(f"    Safety Violations: {sum([o['safety_violations'] for o in severity_outcomes])}")


def demonstrate_demographic_effects():
    """Demonstrate how demographics affect outcomes."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Demographic Effects on Treatment Response")
    print("="*70)
    
    simulator = PatientSimulator(seed=42)
    
    # Generate cohort
    cohort = simulator.generate_diabetes_cohort(n_patients=100)
    
    # Categorize by age groups
    age_groups = {
        'young': [p for p in cohort if p.demographics.age < 40],
        'middle': [p for p in cohort if 40 <= p.demographics.age < 65],
        'elderly': [p for p in cohort if p.demographics.age >= 65]
    }
    
    print("\nAge Group Analysis:")
    for group_name, group_patients in age_groups.items():
        if not group_patients:
            continue
        
        print(f"\n{group_name.upper()} (n={len(group_patients)}):")
        
        # Insulin sensitivity
        p3_values = [p.bergman_params.p3 for p in group_patients]
        print(f"  Insulin Sensitivity (p3): {np.mean(p3_values):.8f}")
        
        # Baseline adherence
        adherence_values = [p.clinical.baseline_adherence for p in group_patients]
        print(f"  Baseline Adherence: {np.mean(adherence_values):.2%}")
        
        # Comorbidities
        total_comorbidities = sum(len(p.clinical.comorbidities) for p in group_patients)
        print(f"  Avg Comorbidities: {total_comorbidities / len(group_patients):.1f}")
    
    # Categorize by BMI
    bmi_groups = {
        'normal': [p for p in cohort if p.demographics.bmi < 25],
        'overweight': [p for p in cohort if 25 <= p.demographics.bmi < 30],
        'obese': [p for p in cohort if p.demographics.bmi >= 30]
    }
    
    print("\n\nBMI Group Analysis:")
    for group_name, group_patients in bmi_groups.items():
        if not group_patients:
            continue
        
        print(f"\n{group_name.upper()} (n={len(group_patients)}):")
        
        # Insulin sensitivity
        p3_values = [p.bergman_params.p3 for p in group_patients]
        print(f"  Insulin Sensitivity (p3): {np.mean(p3_values):.8f}")
        
        # Initial glucose
        glucose_values = [p.initial_glucose for p in group_patients]
        print(f"  Initial Glucose: {np.mean(glucose_values):.1f} mg/dL")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PATIENT SIMULATOR EXAMPLES")
    print("="*70)
    
    # Run all examples
    generate_diabetes_cohort_example()
    generate_adherence_cohort_example()
    test_patient_variability()
    simulate_population_outcomes()
    demonstrate_demographic_effects()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")
