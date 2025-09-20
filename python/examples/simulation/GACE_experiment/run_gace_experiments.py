#!/usr/bin/env python3
"""
GACE Experiments Runner Script

This script provides a convenient way to run all GACE experiments
with different configurations and options.

Usage:
    python run_gace_experiments.py [options]

Options:
    --exp1          Run Experiment 1 (Social utility analysis)
    --exp2          Run Experiment 2 (Cloud server utility analysis)
    --exp3          Run Experiment 3 (Real dataset accuracy/loss)
    --exp4          Run Experiment 4 (Low-quality client analysis)
    --all           Run all experiments
    --dataset       Specify dataset (mnist, cifar10, femnist, svhn)
    --clients       Number of clients (default: 20)
    --edges         Number of edge servers (default: 5)
    --rounds        Number of communication rounds (default: 100)
    --help          Show this help message

Author: GACE Implementation
Date: 2024
"""

import argparse
import sys
import os
import time
import logging
import subprocess
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gace_experiments.log')
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(exp_name: str, dataset: str = None, **kwargs):
    """Run a specific experiment"""
    logger.info(f"Running {exp_name}...")
    
    start_time = time.time()
    
    try:
        if exp_name == "exp1":
            # Run Experiment 1: Social utility analysis
            result = subprocess.run([sys.executable, 'exp1.py'], 
                                  capture_output=True, text=True, cwd=os.path.dirname(__file__))
        elif exp_name == "exp2":
            # Run Experiment 2: Cloud server utility analysis
            result = subprocess.run([sys.executable, 'exp2.py'], 
                                  capture_output=True, text=True, cwd=os.path.dirname(__file__))
        elif exp_name == "exp3":
            # Run Experiment 3: Real dataset experiments
            if dataset:
                exp_file = f'exp3{dataset}.py'
                if os.path.exists(exp_file):
                    result = subprocess.run([sys.executable, exp_file], 
                                          capture_output=True, text=True, cwd=os.path.dirname(__file__))
                else:
                    logger.error(f"Experiment file {exp_file} not found")
                    return False
            else:
                # Run all exp3 experiments
                datasets = ['mnist', 'cifar10', 'femnist', 'svhn']
                for ds in datasets:
                    exp_file = f'exp3{ds}.py'
                    if os.path.exists(exp_file):
                        logger.info(f"Running {exp_file}...")
                        result = subprocess.run([sys.executable, exp_file], 
                                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
                        if result.returncode != 0:
                            logger.error(f"Error running {exp_file}: {result.stderr}")
                    else:
                        logger.warning(f"Experiment file {exp_file} not found")
        elif exp_name == "exp4":
            # Run Experiment 4: Low-quality client analysis
            if dataset:
                exp_file = f'exp4{dataset}.py'
                if os.path.exists(exp_file):
                    result = subprocess.run([sys.executable, exp_file], 
                                          capture_output=True, text=True, cwd=os.path.dirname(__file__))
                else:
                    logger.error(f"Experiment file {exp_file} not found")
                    return False
            else:
                # Run all exp4 experiments
                datasets = ['mnist', 'cifar10', 'femnist', 'svhn']
                for ds in datasets:
                    exp_file = f'exp4{ds}.py'
                    if os.path.exists(exp_file):
                        logger.info(f"Running {exp_file}...")
                        result = subprocess.run([sys.executable, exp_file], 
                                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
                        if result.returncode != 0:
                            logger.error(f"Error running {exp_file}: {result.stderr}")
                    else:
                        logger.warning(f"Experiment file {exp_file} not found")
        else:
            logger.error(f"Unknown experiment: {exp_name}")
            return False
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"{exp_name} completed successfully in {execution_time:.2f} seconds")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            logger.error(f"{exp_name} failed with return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error running {exp_name}: {e}")
        return False

def run_individual_algorithm_test(algorithm: str, dataset: str = "mnist"):
    """Run individual algorithm test"""
    logger.info(f"Testing {algorithm} algorithm on {dataset} dataset...")
    
    config_file = f"selected_{dataset}.yaml"
    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} not found")
        return False
    
    algorithm_file = f"gace_{algorithm.lower()}_torch_hierarchicalfl_step_by_step_exp.py"
    if not os.path.exists(algorithm_file):
        logger.error(f"Algorithm file {algorithm_file} not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, algorithm_file, '--cf', config_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            logger.info(f"{algorithm} test completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            logger.error(f"{algorithm} test failed with return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error testing {algorithm}: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("Checking dependencies...")
    
    required_files = [
        'gace_algorithm.py',
        'gace_torch_hierarchicalfl_step_by_step_exp.py',
        'gace_ra_torch_hierarchicalfl_step_by_step_exp.py',
        'gace_no_torch_hierarchicalfl_step_by_step_exp.py',
        'exp1.py',
        'exp2.py',
        'exp3mnist.py',
        'exp3cifar10.py',
        'selected_mnist.yaml',
        'selected_cifar10.yaml',
        'selected_femnist.yaml',
        'selected_svhn.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    # Check Python packages
    try:
        import numpy
        import matplotlib
        import scipy
        import fedml
        logger.info("All required Python packages are available")
        return True
    except ImportError as e:
        logger.error(f"Missing required Python package: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GACE Experiments Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gace_experiments.py --exp1                    # Run Experiment 1
  python run_gace_experiments.py --exp3 --dataset mnist    # Run Experiment 3 on MNIST
  python run_gace_experiments.py --all                     # Run all experiments
  python run_gace_experiments.py --test gace --dataset cifar10  # Test GACE on CIFAR-10
        """
    )
    
    # Experiment options
    parser.add_argument('--exp1', action='store_true',
                       help='Run Experiment 1 (Social utility analysis)')
    parser.add_argument('--exp2', action='store_true',
                       help='Run Experiment 2 (Cloud server utility analysis)')
    parser.add_argument('--exp3', action='store_true',
                       help='Run Experiment 3 (Real dataset accuracy/loss)')
    parser.add_argument('--exp4', action='store_true',
                       help='Run Experiment 4 (Low-quality client analysis)')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    
    # Test options
    parser.add_argument('--test', type=str, choices=['gace', 'ra', 'gace-no'],
                       help='Test individual algorithm')
    
    # Configuration options
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'femnist', 'svhn'],
                       help='Dataset for experiments')
    parser.add_argument('--clients', type=int, default=20,
                       help='Number of clients (default: 20)')
    parser.add_argument('--edges', type=int, default=5,
                       help='Number of edge servers (default: 5)')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of communication rounds (default: 100)')
    
    # Utility options
    parser.add_argument('--check', action='store_true',
                       help='Check dependencies and file structure')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    try:
        # Check dependencies if requested
        if args.check:
            if check_dependencies():
                print("✓ All dependencies are satisfied")
            else:
                print("✗ Some dependencies are missing")
                sys.exit(1)
            return
        
        # Run individual algorithm test
        if args.test:
            if not args.dataset:
                args.dataset = 'mnist'  # Default dataset
            success = run_individual_algorithm_test(args.test, args.dataset)
            if not success:
                sys.exit(1)
            return
        
        # Run experiments
        if args.all:
            # Run all experiments
            experiments = ['exp1', 'exp2', 'exp3', 'exp4']
            for exp in experiments:
                success = run_experiment(exp, args.dataset)
                if not success:
                    logger.warning(f"Experiment {exp} failed, continuing with next experiment...")
        else:
            # Run specific experiments
            if args.exp1:
                run_experiment('exp1', args.dataset)
            if args.exp2:
                run_experiment('exp2', args.dataset)
            if args.exp3:
                run_experiment('exp3', args.dataset)
            if args.exp4:
                run_experiment('exp4', args.dataset)
        
        print("\n" + "="*80)
        print("GACE EXPERIMENTS COMPLETED!")
        print("="*80)
        print("Check the generated PNG files for results:")
        print("- exp1_social_utility_results.png")
        print("- exp2_cloud_utility_results.png")
        print("- exp3_*_results.png")
        print("- exp4_*_results.png")
        print("\nLog file: gace_experiments.log")
        
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

