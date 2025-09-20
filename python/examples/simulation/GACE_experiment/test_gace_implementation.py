#!/usr/bin/env python3
"""
GACE Implementation Test Script

This script tests the basic functionality of the GACE implementation
to ensure all components are working correctly.

Author: GACE Implementation
Date: 2024
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gace_algorithm():
    """Test the core GACE algorithm"""
    logger.info("Testing GACE algorithm...")
    
    try:
        from gace_algorithm import GACEAlgorithm, GACEParameters
        
        # Create parameters
        params = GACEParameters(M=3, N=6)  # Small system for testing
        
        # Create and run GACE algorithm
        gace = GACEAlgorithm(params)
        results = gace.execute(seed=42)
        
        # Check results
        assert results.U_social > 0, "Social utility should be positive"
        assert len(results.Pi_star) == params.M, "Should have M clusters"
        assert results.S_star.shape == (params.N, params.M), "Association matrix should be NxM"
        assert results.P_star > 0, "Service price should be positive"
        assert len(results.Gamma_star) == params.M, "Should have M edge server rewards"
        assert len(results.D_star) == params.N, "Should have N client data plans"
        
        logger.info("✓ GACE algorithm test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ GACE algorithm test failed: {e}")
        return False

def test_gace_experiment():
    """Test the GACE experiment runner"""
    logger.info("Testing GACE experiment runner...")
    
    try:
        from gace_torch_hierarchicalfl_step_by_step_exp import (
            initialize_gace_system, execute_gace_algorithm, 
            calculate_social_utility, reset_gace_system
        )
        
        # Create mock args
        class Args:
            def __init__(self):
                self.group_num = 3
                self.client_num_in_total = 6
                self.random_seed = 42
                self.low_quality_ratio = 0.0
        
        args = Args()
        
        # Test initialization
        initialize_gace_system(args)
        
        # Test execution
        results = execute_gace_algorithm()
        
        # Test utility calculation
        social_utility = calculate_social_utility()
        assert social_utility > 0, "Social utility should be positive"
        
        # Test reset
        reset_gace_system()
        
        logger.info("✓ GACE experiment runner test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ GACE experiment runner test failed: {e}")
        return False

def test_ra_algorithm():
    """Test the RA (Random Association) algorithm"""
    logger.info("Testing RA algorithm...")
    
    try:
        from gace_ra_torch_hierarchicalfl_step_by_step_exp import (
            initialize_ra_system, execute_ra_algorithm,
            calculate_social_utility, reset_gace_system
        )
        
        # Create mock args
        class Args:
            def __init__(self):
                self.group_num = 3
                self.client_num_in_total = 6
                self.random_seed = 42
                self.low_quality_ratio = 0.0
        
        args = Args()
        
        # Test initialization
        initialize_ra_system(args)
        
        # Test execution
        results = execute_ra_algorithm()
        
        # Test utility calculation
        social_utility = calculate_social_utility()
        assert social_utility > 0, "Social utility should be positive"
        
        # Test reset
        reset_gace_system()
        
        logger.info("✓ RA algorithm test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ RA algorithm test failed: {e}")
        return False

def test_gace_no_algorithm():
    """Test the GACE-NO algorithm"""
    logger.info("Testing GACE-NO algorithm...")
    
    try:
        from gace_no_torch_hierarchicalfl_step_by_step_exp import (
            initialize_gace_no_system, execute_gace_no_algorithm,
            calculate_social_utility, reset_gace_system
        )
        
        # Create mock args
        class Args:
            def __init__(self):
                self.group_num = 3
                self.client_num_in_total = 6
                self.random_seed = 42
                self.low_quality_ratio = 0.0
        
        args = Args()
        
        # Test initialization
        initialize_gace_no_system(args)
        
        # Test execution
        results = execute_gace_no_algorithm()
        
        # Test utility calculation
        social_utility = calculate_social_utility()
        assert social_utility > 0, "Social utility should be positive"
        
        # Test reset
        reset_gace_system()
        
        logger.info("✓ GACE-NO algorithm test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ GACE-NO algorithm test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration files"""
    logger.info("Testing configuration files...")
    
    config_files = [
        'selected_mnist.yaml',
        'selected_cifar10.yaml', 
        'selected_femnist.yaml',
        'selected_svhn.yaml'
    ]
    
    try:
        import yaml
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check required sections
                assert 'data_args' in config, f"{config_file} missing data_args"
                assert 'model_args' in config, f"{config_file} missing model_args"
                assert 'train_args' in config, f"{config_file} missing train_args"
                
                logger.info(f"✓ {config_file} is valid")
            else:
                logger.warning(f"⚠ {config_file} not found")
        
        logger.info("✓ Configuration files test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration files test failed: {e}")
        return False

def test_experiment_files():
    """Test experiment files"""
    logger.info("Testing experiment files...")
    
    experiment_files = [
        'exp1.py',
        'exp2.py', 
        'exp3mnist.py',
        'exp3cifar10.py'
    ]
    
    try:
        for exp_file in experiment_files:
            if os.path.exists(exp_file):
                # Try to import the file to check for syntax errors
                spec = __import__(exp_file[:-3], fromlist=[''])
                logger.info(f"✓ {exp_file} imports successfully")
            else:
                logger.warning(f"⚠ {exp_file} not found")
        
        logger.info("✓ Experiment files test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Experiment files test failed: {e}")
        return False

def run_performance_comparison():
    """Run a quick performance comparison"""
    logger.info("Running performance comparison...")
    
    try:
        from gace_torch_hierarchicalfl_step_by_step_exp import (
            initialize_gace_system, execute_gace_algorithm, 
            calculate_social_utility, reset_gace_system
        )
        from gace_ra_torch_hierarchicalfl_step_by_step_exp import (
            initialize_ra_system, execute_ra_algorithm,
            calculate_social_utility as calculate_ra_social_utility, 
            reset_gace_system as reset_ra_system
        )
        from gace_no_torch_hierarchicalfl_step_by_step_exp import (
            initialize_gace_no_system, execute_gace_no_algorithm,
            calculate_social_utility as calculate_gace_no_social_utility,
            reset_gace_system as reset_gace_no_system
        )
        
        # Create mock args
        class Args:
            def __init__(self):
                self.group_num = 5
                self.client_num_in_total = 20
                self.random_seed = 42
                self.low_quality_ratio = 0.0
        
        args = Args()
        
        # Test GACE
        initialize_gace_system(args)
        execute_gace_algorithm()
        gace_utility = calculate_social_utility()
        reset_gace_system()
        
        # Test RA
        initialize_ra_system(args)
        execute_ra_algorithm()
        ra_utility = calculate_ra_social_utility()
        reset_ra_system()
        
        # Test GACE-NO
        initialize_gace_no_system(args)
        execute_gace_no_algorithm()
        gace_no_utility = calculate_gace_no_social_utility()
        reset_gace_no_system()
        
        # Print comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"GACE Social Utility:     {gace_utility:.4f}")
        print(f"RA Social Utility:       {ra_utility:.4f}")
        print(f"GACE-NO Social Utility:  {gace_no_utility:.4f}")
        print("="*60)
        
        # Check if GACE performs better
        if gace_utility > ra_utility and gace_utility > gace_no_utility:
            logger.info("✓ GACE performs better than baselines")
            return True
        else:
            logger.warning("⚠ GACE does not perform better than baselines")
            return False
        
    except Exception as e:
        logger.error(f"✗ Performance comparison failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*80)
    print("GACE IMPLEMENTATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("GACE Algorithm", test_gace_algorithm),
        ("GACE Experiment Runner", test_gace_experiment),
        ("RA Algorithm", test_ra_algorithm),
        ("GACE-NO Algorithm", test_gace_no_algorithm),
        ("Configuration Files", test_configuration_files),
        ("Experiment Files", test_experiment_files),
        ("Performance Comparison", run_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! GACE implementation is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

