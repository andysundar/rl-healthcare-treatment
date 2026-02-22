"""
Configuration Template for Evaluation Framework

Copy this file and customize for your specific evaluation needs.

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

# Import configuration classes
from src.evaluation import (
    EvaluationFrameworkConfig,
    OPEConfig,
    SafetyConfig,
    ClinicalConfig,
    PerformanceConfig
)


# ============================================================================
# CUSTOM CONFIGURATION FOR YOUR THESIS
# ============================================================================

class ThesisEvaluationConfig:
    """
    Configuration optimized for M.Tech thesis evaluation.
    
    Focuses on:
    - Safety-constrained RL validation
    - Clinical outcome metrics
    - MIMIC-III dataset evaluation
    - Conservative Q-Learning (CQL) assessment
    """
    
    @staticmethod
    def get_cql_evaluation_config():
        """Configuration for CQL policy evaluation"""
        
        # OPE Configuration
        ope_config = OPEConfig(
            gamma=0.99,
            clip_importance_ratio=True,
            max_importance_ratio=100.0,
            min_importance_ratio=0.01,
            confidence_level=0.95,
            n_bootstrap_samples=1000
        )
        
        # Safety Configuration (Diabetes-focused)
        safety_config = SafetyConfig(
            # Glucose safety ranges (mg/dL)
            safe_glucose_min=70.0,
            safe_glucose_max=180.0,
            critical_hypoglycemia=54.0,
            critical_hyperglycemia=250.0,
            
            # Blood pressure safety ranges (mmHg)
            safe_bp_systolic_min=90.0,
            safe_bp_systolic_max=140.0,
            safe_bp_diastolic_min=60.0,
            safe_bp_diastolic_max=90.0,
            
            # HbA1c safety range (%)
            safe_hba1c_min=5.0,
            safe_hba1c_max=7.0,
            
            # Time-based settings
            max_time_in_unsafe_state=3,  # consecutive timesteps
        )
        
        # Clinical Configuration
        clinical_config = ClinicalConfig(
            # Target ranges (stricter than safety)
            target_glucose_min=80.0,
            target_glucose_max=130.0,
            target_bp_systolic_min=90.0,
            target_bp_systolic_max=120.0,
            target_hba1c_min=5.0,
            target_hba1c_max=6.5,
            
            # Adherence thresholds
            good_adherence_threshold=0.80,
            excellent_adherence_threshold=0.90,
        )
        
        # Performance Configuration
        performance_config = PerformanceConfig(
            success_threshold=0.8,
            confidence_level=0.95,
            n_bootstrap_samples=1000
        )
        
        # Master Configuration
        return EvaluationFrameworkConfig(
            ope_config=ope_config,
            safety_config=safety_config,
            clinical_config=clinical_config,
            performance_config=performance_config,
            output_dir='./evaluation_results/cql_mimic',
            save_detailed_results=True,
            save_visualizations=True,
            ope_methods=['IS', 'WIS', 'DR', 'DM'],
            run_safety_evaluation=True,
            run_clinical_evaluation=True,
            run_performance_evaluation=True,
            run_ope_evaluation=True
        )
    
    @staticmethod
    def get_comparison_config():
        """Configuration for comparing multiple policies"""
        
        config = ThesisEvaluationConfig.get_cql_evaluation_config()
        config.output_dir = './evaluation_results/policy_comparison'
        return config
    
    @staticmethod
    def get_ablation_study_config():
        """Configuration for ablation studies"""
        
        config = ThesisEvaluationConfig.get_cql_evaluation_config()
        config.output_dir = './evaluation_results/ablation_study'
        # For ablation, you might want to disable some evaluations to save time
        config.run_ope_evaluation = False  # Skip OPE for faster ablation
        return config
    
    @staticmethod
    def get_transfer_learning_config():
        """Configuration for transfer learning evaluation"""
        
        config = ThesisEvaluationConfig.get_cql_evaluation_config()
        config.output_dir = './evaluation_results/transfer_learning'
        
        # Might want to adjust safety/clinical ranges for target population
        # For example, if transferring to Indian population:
        # config.safety_config.safe_glucose_min = 72.0  # Adjust based on population
        
        return config


# ============================================================================
# YAML EXPORT/IMPORT
# ============================================================================

def save_config_to_yaml(config: EvaluationFrameworkConfig, filepath: str):
    """Save configuration to YAML file"""
    
    config_dict = {
        'ope_config': {
            'gamma': config.ope_config.gamma,
            'clip_importance_ratio': config.ope_config.clip_importance_ratio,
            'max_importance_ratio': config.ope_config.max_importance_ratio,
            'min_importance_ratio': config.ope_config.min_importance_ratio,
            'confidence_level': config.ope_config.confidence_level,
            'n_bootstrap_samples': config.ope_config.n_bootstrap_samples
        },
        'safety_config': {
            'safe_glucose_min': config.safety_config.safe_glucose_min,
            'safe_glucose_max': config.safety_config.safe_glucose_max,
            'critical_hypoglycemia': config.safety_config.critical_hypoglycemia,
            'critical_hyperglycemia': config.safety_config.critical_hyperglycemia,
            'safe_bp_systolic_min': config.safety_config.safe_bp_systolic_min,
            'safe_bp_systolic_max': config.safety_config.safe_bp_systolic_max,
            'safe_hba1c_min': config.safety_config.safe_hba1c_min,
            'safe_hba1c_max': config.safety_config.safe_hba1c_max
        },
        'clinical_config': {
            'target_glucose_min': config.clinical_config.target_glucose_min,
            'target_glucose_max': config.clinical_config.target_glucose_max,
            'target_bp_systolic_min': config.clinical_config.target_bp_systolic_min,
            'target_bp_systolic_max': config.clinical_config.target_bp_systolic_max,
            'target_hba1c_min': config.clinical_config.target_hba1c_min,
            'target_hba1c_max': config.clinical_config.target_hba1c_max
        },
        'evaluation_settings': {
            'output_dir': config.output_dir,
            'ope_methods': config.ope_methods,
            'run_safety_evaluation': config.run_safety_evaluation,
            'run_clinical_evaluation': config.run_clinical_evaluation,
            'run_performance_evaluation': config.run_performance_evaluation,
            'run_ope_evaluation': config.run_ope_evaluation,
            'save_visualizations': config.save_visualizations
        }
    }
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"Configuration saved to {filepath}")


def load_config_from_yaml(filepath: str) -> EvaluationFrameworkConfig:
    """Load configuration from YAML file"""
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Reconstruct configuration objects
    ope_config = OPEConfig(**config_dict['ope_config'])
    safety_config = SafetyConfig(**config_dict['safety_config'])
    clinical_config = ClinicalConfig(**config_dict['clinical_config'])
    
    eval_settings = config_dict['evaluation_settings']
    
    return EvaluationFrameworkConfig(
        ope_config=ope_config,
        safety_config=safety_config,
        clinical_config=clinical_config,
        **eval_settings
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Evaluation Configuration Templates")
    print("=" * 80)
    
    # 1. Create CQL evaluation config
    print("\n1. Creating CQL evaluation configuration...")
    cql_config = ThesisEvaluationConfig.get_cql_evaluation_config()
    save_config_to_yaml(cql_config, './configs/cql_evaluation.yaml')
    
    # 2. Create comparison config
    print("\n2. Creating policy comparison configuration...")
    comparison_config = ThesisEvaluationConfig.get_comparison_config()
    save_config_to_yaml(comparison_config, './configs/comparison.yaml')
    
    # 3. Create ablation study config
    print("\n3. Creating ablation study configuration...")
    ablation_config = ThesisEvaluationConfig.get_ablation_study_config()
    save_config_to_yaml(ablation_config, './configs/ablation.yaml')
    
    # 4. Create transfer learning config
    print("\n4. Creating transfer learning configuration...")
    transfer_config = ThesisEvaluationConfig.get_transfer_learning_config()
    save_config_to_yaml(transfer_config, './configs/transfer_learning.yaml')
    
    print("\n" + "=" * 80)
    print("Configuration files created in ./configs/")
    print("\nTo use in your code:")
    print("  config = load_config_from_yaml('./configs/cql_evaluation.yaml')")
    print("  evaluator = ComprehensiveEvaluator(config=config)")
