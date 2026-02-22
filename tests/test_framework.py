"""
Test Script for Evaluation Framework

Verifies all components work correctly before integration with main codebase.

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.evaluation.off_policy_eval import (
            OPEEvaluator, ImportanceSampling, WeightedImportanceSampling,
            DoublyRobust, DirectMethod, OPEConfig
        )
        print("  ✓ off_policy_eval imports successful")
    except Exception as e:
        print(f"  ✗ off_policy_eval import failed: {e}")
        return False
    
    try:
        from src.evaluation.safety_metrics import (
            SafetyEvaluator, SafeStateChecker, SafetyConfig
        )
        print("  ✓ safety_metrics imports successful")
    except Exception as e:
        print(f"  ✗ safety_metrics import failed: {e}")
        return False
    
    try:
        from src.evaluation.clinical_metrics import (
            ClinicalEvaluator, ClinicalConfig
        )
        print("  ✓ clinical_metrics imports successful")
    except Exception as e:
        print(f"  ✗ clinical_metrics import failed: {e}")
        return False
    
    try:
        from src.evaluation.performance_metrics import (
            PerformanceMetrics, PolicyComparator, PerformanceConfig
        )
        print("  ✓ performance_metrics imports successful")
    except Exception as e:
        print(f"  ✗ performance_metrics import failed: {e}")
        return False
    
    try:
        from src.evaluation.visualizations import EvaluationVisualizer
        print("  ✓ visualizations imports successful")
    except Exception as e:
        print(f"  ✗ visualizations import failed: {e}")
        return False
    
    try:
        from src.evaluation import (
            ComprehensiveEvaluator, EvaluationFrameworkConfig
        )
        print("  ✓ main evaluation module imports successful")
    except Exception as e:
        print(f"  ✗ main evaluation module import failed: {e}")
        return False
    
    return True


def generate_test_trajectories(n_episodes=50, min_length=20, max_length=100):
    """Generate synthetic trajectories for testing"""
    trajectories = []
    
    for _ in range(n_episodes):
        episode_length = np.random.randint(min_length, max_length)
        trajectory = []
        
        for t in range(episode_length):
            # Generate realistic-looking patient state
            state = {
                'glucose': np.random.uniform(60, 200),
                'blood_pressure_systolic': np.random.uniform(80, 160),
                'blood_pressure_diastolic': np.random.uniform(50, 100),
                'hba1c': np.random.uniform(4.5, 8.5),
                'adherence_score': np.random.uniform(0.3, 1.0),
                'cholesterol': np.random.uniform(150, 250),
                'ldl': np.random.uniform(70, 150),
                'hdl': np.random.uniform(30, 70)
            }
            
            action = np.random.randn(5)  # 5-dim action space
            reward = np.random.randn()  # Random reward
            next_state = state.copy()  # Simplified
            done = (t == episode_length - 1)
            
            trajectory.append((state, action, reward, next_state, done))
        
        trajectories.append(trajectory)
    
    return trajectories


class MockPolicy:
    """Mock policy for testing"""
    def select_action(self, state, deterministic=True):
        return np.random.randn(5)
    
    def get_action_probability(self, state, action):
        return 0.5
    
    def sample_action(self, state):
        return self.select_action(state, deterministic=False)


class MockQFunction:
    """Mock Q-function for testing"""
    def predict(self, state, action):
        return np.random.randn()


def test_ope_evaluator():
    """Test OPE evaluator"""
    print("\nTesting OPE Evaluator...")
    
    from src.evaluation.off_policy_eval import OPEEvaluator, OPEConfig
    
    # Create mock components
    policy = MockPolicy()
    behavior_policy = MockPolicy()
    q_function = MockQFunction()
    trajectories = generate_test_trajectories(20)
    
    # Initialize evaluator
    config = OPEConfig()
    evaluator = OPEEvaluator(q_function=q_function, config=config)
    
    try:
        # Test all methods
        results = evaluator.evaluate_all(
            policy,
            behavior_policy,
            trajectories,
            methods=['IS', 'WIS', 'DR', 'DM']
        )
        
        print(f"  ✓ OPE evaluation successful")
        print(f"    Methods tested: {list(results.keys())}")
        for method, result in results.items():
            print(f"    {method}: mean={result['mean']:.3f}, std={result.get('std', 'N/A')}")
        
        # Test comparison
        comparison = evaluator.compare_estimates(results)
        print(f"  ✓ OPE comparison successful")
        print(f"    Mean of means: {comparison['mean_of_means']:.3f}")
        print(f"    Agreement std: {comparison['agreement_std']:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ OPE evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safety_evaluator():
    """Test safety evaluator"""
    print("\nTesting Safety Evaluator...")
    
    from src.evaluation.safety_metrics import SafetyEvaluator, SafetyConfig
    
    config = SafetyConfig()
    evaluator = SafetyEvaluator(config=config)
    trajectories = generate_test_trajectories(30)
    
    try:
        results = evaluator.evaluate_comprehensive(trajectories)
        
        print(f"  ✓ Safety evaluation successful")
        print(f"    Safety Index: {results['safety_index_metrics']['safety_index']:.3f}")
        print(f"    Violation Rate: {results['violation_rate_metrics']['violation_rate']:.3f}")
        print(f"    Overall Safety Score: {results['overall_safety_score']:.3f}")
        print(f"    Safety Passed: {results['safety_passed']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Safety evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clinical_evaluator():
    """Test clinical evaluator"""
    print("\nTesting Clinical Evaluator...")
    
    from src.evaluation.clinical_metrics import ClinicalEvaluator, ClinicalConfig
    
    config = ClinicalConfig()
    evaluator = ClinicalEvaluator(config=config)
    
    policy_trajectories = generate_test_trajectories(30)
    baseline_trajectories = generate_test_trajectories(30)
    
    try:
        results = evaluator.evaluate_comprehensive(
            policy_trajectories,
            baseline_trajectories
        )
        
        print(f"  ✓ Clinical evaluation successful")
        print(f"    Overall Clinical Score: {results['overall_clinical_score']:.3f}")
        print(f"    Adverse Event Rate: {results['adverse_events']['adverse_event_rate']:.3f}")
        
        # Time in range
        for metric, tir_data in results.get('time_in_range', {}).items():
            if 'time_in_range' in tir_data:
                print(f"    {metric} TIR: {tir_data['time_in_range']*100:.1f}%")
        
        return True
    except Exception as e:
        print(f"  ✗ Clinical evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_metrics():
    """Test performance metrics"""
    print("\nTesting Performance Metrics...")
    
    from src.evaluation.performance_metrics import PerformanceMetrics, PerformanceConfig
    
    config = PerformanceConfig()
    evaluator = PerformanceMetrics(config=config)
    trajectories = generate_test_trajectories(30)
    
    try:
        # Test returns
        returns = evaluator.compute_average_return(trajectories)
        print(f"  ✓ Return computation successful")
        print(f"    Mean Return: {returns['mean_return']:.3f}")
        
        # Test success rate
        success = evaluator.compute_success_rate(trajectories)
        print(f"  ✓ Success rate computation successful")
        print(f"    Success Rate: {success['success_rate']:.3f}")
        
        # Test episode lengths
        lengths = evaluator.compute_episode_lengths(trajectories)
        print(f"  ✓ Episode length computation successful")
        print(f"    Mean Length: {lengths['mean_length']:.1f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Performance metrics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_comparator():
    """Test policy comparator"""
    print("\nTesting Policy Comparator...")
    
    from src.evaluation.performance_metrics import PolicyComparator, PerformanceConfig
    
    config = PerformanceConfig()
    comparator = PolicyComparator(config=config)
    
    policies_dict = {
        'Policy_A': generate_test_trajectories(20),
        'Policy_B': generate_test_trajectories(20),
        'Policy_C': generate_test_trajectories(20)
    }
    
    try:
        comparison_df = comparator.compare_policies(
            policies_dict,
            baseline_name='Policy_C'
        )
        
        print(f"  ✓ Policy comparison successful")
        print(f"    Policies compared: {len(comparison_df)}")
        print(f"    Metrics computed: {len(comparison_df.columns)}")
        print(f"\n{comparison_df[['mean_return', 'success_rate']].to_string()}")
        
        return True
    except Exception as e:
        print(f"  ✗ Policy comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizations():
    """Test visualization module"""
    print("\nTesting Visualizations...")
    
    from src.evaluation.visualizations import EvaluationVisualizer
    from src.evaluation.safety_metrics import SafeStateChecker, SafetyConfig
    import tempfile
    
    # Create temporary directory for plots
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = EvaluationVisualizer(output_dir=tmpdir)
        
        try:
            # Test learning curves
            training_curves = {
                'Policy_A': np.random.randn(500).cumsum().tolist(),
                'Policy_B': np.random.randn(500).cumsum().tolist()
            }
            viz.plot_learning_curves(training_curves)
            print(f"  ✓ Learning curves plot successful")
            
            # Test safety violations
            trajectories = generate_test_trajectories(20)
            safe_state_checker = SafeStateChecker(SafetyConfig())
            viz.plot_safety_violations(trajectories, safe_state_checker)
            print(f"  ✓ Safety violations plot successful")
            
            # Test health metrics
            viz.plot_health_metrics_trajectory(trajectories)
            print(f"  ✓ Health metrics plot successful")
            
            return True
        except Exception as e:
            print(f"  ✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_comprehensive_evaluator():
    """Test comprehensive evaluator integration"""
    print("\nTesting Comprehensive Evaluator...")
    
    from src.evaluation import ComprehensiveEvaluator, EvaluationFrameworkConfig
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EvaluationFrameworkConfig(
            output_dir=tmpdir,
            save_visualizations=False,  # Skip viz in test
            run_ope_evaluation=False  # Skip OPE to avoid requiring behavior policy
        )
        
        policy = MockPolicy()
        trajectories = generate_test_trajectories(30)
        
        try:
            evaluator = ComprehensiveEvaluator(config=config)
            
            results = evaluator.evaluate_policy(
                policy_name='Test_Policy',
                policy=policy,
                policy_trajectories=trajectories
            )
            
            print(f"  ✓ Comprehensive evaluation successful")
            print(f"    Summary keys: {list(results['summary'].keys())}")
            print(f"    Safety Index: {results['summary'].get('safety_index', 'N/A')}")
            print(f"    Overall Safety: {results['summary'].get('overall_safety_score', 'N/A')}")
            print(f"    Overall Clinical: {results['summary'].get('overall_clinical_score', 'N/A')}")
            
            return True
        except Exception as e:
            print(f"  ✗ Comprehensive evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("EVALUATION FRAMEWORK TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("OPE Evaluator", test_ope_evaluator),
        ("Safety Evaluator", test_safety_evaluator),
        ("Clinical Evaluator", test_clinical_evaluator),
        ("Performance Metrics", test_performance_metrics),
        ("Policy Comparator", test_policy_comparator),
        ("Visualizations", test_visualizations),
        ("Comprehensive Evaluator", test_comprehensive_evaluator)
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Framework is ready for integration.")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
