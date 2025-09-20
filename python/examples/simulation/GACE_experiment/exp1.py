"""
GACE Experiment 1: Social Utility vs Number of Clients and Edge Servers (without low-quality clients)
This experiment compares GACE, RA, and GACE-NO on synthetic dataset focusing on social utility
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import logging
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from gace_torch_hierarchicalfl_step_by_step_exp import (
    initialize_gace_system, execute_gace_algorithm, calculate_social_utility,
    reset_gace_system
)
from gace_ra_torch_hierarchicalfl_step_by_step_exp import (
    initialize_ra_system, execute_ra_algorithm, calculate_social_utility as calculate_ra_social_utility,
    reset_gace_system as reset_ra_system
)
from gace_no_torch_hierarchicalfl_step_by_step_exp import (
    initialize_gace_no_system, execute_gace_no_algorithm, calculate_social_utility as calculate_gace_no_social_utility,
    reset_gace_system as reset_gace_no_system
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for Experiment 1"""
    def __init__(self):
        # Client number range
        self.client_range = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Edge server number range  
        self.edge_range = [5, 10, 15, 20, 25, 30, 35]
        # Number of runs for averaging
        self.num_runs = 5
        # Random seed base
        self.seed_base = 42
        # Low quality ratio (0.0 for this experiment)
        self.low_quality_ratio = 0.0

def run_social_utility_vs_clients():
    """Run experiment: Social utility vs number of clients"""
    logger.info("Running Experiment 1a: Social Utility vs Number of Clients")
    
    config = ExperimentConfig()
    results = {
        'GACE': [],
        'RA': [],
        'GACE-NO': []
    }
    
    for num_clients in config.client_range:
        logger.info(f"Testing with {num_clients} clients...")
        
        gace_utilities = []
        ra_utilities = []
        gace_no_utilities = []
        
        for run in range(config.num_runs):
            seed = config.seed_base + run
            
            # Test GACE
            try:
                # Create mock args
                class Args:
                    def __init__(self):
                        self.group_num = 5
                        self.client_num_in_total = num_clients
                        self.random_seed = seed
                        self.low_quality_ratio = config.low_quality_ratio
                
                args = Args()
                
                # Run GACE
                initialize_gace_system(args)
                execute_gace_algorithm()
                gace_utility = calculate_social_utility()
                gace_utilities.append(gace_utility)
                reset_gace_system()
                
                # Run RA
                initialize_ra_system(args)
                execute_ra_algorithm()
                ra_utility = calculate_ra_social_utility()
                ra_utilities.append(ra_utility)
                reset_ra_system()
                
                # Run GACE-NO
                initialize_gace_no_system(args)
                execute_gace_no_algorithm()
                gace_no_utility = calculate_gace_no_social_utility()
                gace_no_utilities.append(gace_no_utility)
                reset_gace_no_system()
                
            except Exception as e:
                logger.error(f"Error in run {run} with {num_clients} clients: {e}")
                continue
        
        # Calculate averages
        if gace_utilities:
            results['GACE'].append(np.mean(gace_utilities))
        else:
            results['GACE'].append(0.0)
            
        if ra_utilities:
            results['RA'].append(np.mean(ra_utilities))
        else:
            results['RA'].append(0.0)
            
        if gace_no_utilities:
            results['GACE-NO'].append(np.mean(gace_no_utilities))
        else:
            results['GACE-NO'].append(0.0)
    
    return results, config.client_range

def run_social_utility_vs_edge_servers():
    """Run experiment: Social utility vs number of edge servers"""
    logger.info("Running Experiment 1b: Social Utility vs Number of Edge Servers")
    
    config = ExperimentConfig()
    results = {
        'GACE': [],
        'RA': [],
        'GACE-NO': []
    }
    
    for num_edges in config.edge_range:
        logger.info(f"Testing with {num_edges} edge servers...")
        
        gace_utilities = []
        ra_utilities = []
        gace_no_utilities = []
        
        for run in range(config.num_runs):
            seed = config.seed_base + run
            
            # Create mock args
            class Args:
                def __init__(self):
                    self.group_num = num_edges
                    self.client_num_in_total = 40  # Fixed number of clients
                    self.random_seed = seed
                    self.low_quality_ratio = config.low_quality_ratio
            
            args = Args()
            
            try:
                # Run GACE
                initialize_gace_system(args)
                execute_gace_algorithm()
                gace_utility = calculate_social_utility()
                gace_utilities.append(gace_utility)
                reset_gace_system()
                
                # Run RA
                initialize_ra_system(args)
                execute_ra_algorithm()
                ra_utility = calculate_ra_social_utility()
                ra_utilities.append(ra_utility)
                reset_ra_system()
                
                # Run GACE-NO
                initialize_gace_no_system(args)
                execute_gace_no_algorithm()
                gace_no_utility = calculate_gace_no_social_utility()
                gace_no_utilities.append(gace_no_utility)
                reset_gace_no_system()
                
            except Exception as e:
                logger.error(f"Error in run {run} with {num_edges} edge servers: {e}")
                continue
        
        # Calculate averages
        if gace_utilities:
            results['GACE'].append(np.mean(gace_utilities))
        else:
            results['GACE'].append(0.0)
            
        if ra_utilities:
            results['RA'].append(np.mean(ra_utilities))
        else:
            results['RA'].append(0.0)
            
        if gace_no_utilities:
            results['GACE-NO'].append(np.mean(gace_no_utilities))
        else:
            results['GACE-NO'].append(0.0)
    
    return results, config.edge_range

def plot_results(results_clients, x_values_clients, results_edges, x_values_edges):
    """Plot the experimental results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Social Utility vs Number of Clients
    ax1.plot(x_values_clients, results_clients['GACE'], 'b-o', label='GACE', linewidth=2, markersize=6)
    ax1.plot(x_values_clients, results_clients['RA'], 'r-s', label='RA', linewidth=2, markersize=6)
    ax1.plot(x_values_clients, results_clients['GACE-NO'], 'g-^', label='GACE-NO', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Clients', fontsize=12)
    ax1.set_ylabel('Social Utility', fontsize=12)
    ax1.set_title('Social Utility vs Number of Clients (No Low-Quality Clients)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Social Utility vs Number of Edge Servers
    ax2.plot(x_values_edges, results_edges['GACE'], 'b-o', label='GACE', linewidth=2, markersize=6)
    ax2.plot(x_values_edges, results_edges['RA'], 'r-s', label='RA', linewidth=2, markersize=6)
    ax2.plot(x_values_edges, results_edges['GACE-NO'], 'g-^', label='GACE-NO', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Edge Servers', fontsize=12)
    ax2.set_ylabel('Social Utility', fontsize=12)
    ax2.set_title('Social Utility vs Number of Edge Servers (No Low-Quality Clients)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp1_social_utility_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("\n" + "="*80)
    print("EXPERIMENT 1 RESULTS: Social Utility (No Low-Quality Clients)")
    print("="*80)
    
    print("\nSocial Utility vs Number of Clients:")
    print("Clients\tGACE\t\tRA\t\tGACE-NO")
    print("-" * 50)
    for i, num_clients in enumerate(x_values_clients):
        print(f"{num_clients}\t{results_clients['GACE'][i]:.4f}\t\t{results_clients['RA'][i]:.4f}\t\t{results_clients['GACE-NO'][i]:.4f}")
    
    print("\nSocial Utility vs Number of Edge Servers:")
    print("Edges\tGACE\t\tRA\t\tGACE-NO")
    print("-" * 50)
    for i, num_edges in enumerate(x_values_edges):
        print(f"{num_edges}\t{results_edges['GACE'][i]:.4f}\t\t{results_edges['RA'][i]:.4f}\t\t{results_edges['GACE-NO'][i]:.4f}")

def main():
    """Main function to run Experiment 1"""
    print("="*80)
    print("GACE EXPERIMENT 1: Social Utility Analysis (No Low-Quality Clients)")
    print("="*80)
    
    start_time = time.time()
    
    # Run experiments
    logger.info("Starting Experiment 1...")
    
    # Experiment 1a: Social utility vs number of clients
    results_clients, x_values_clients = run_social_utility_vs_clients()
    
    # Experiment 1b: Social utility vs number of edge servers
    results_edges, x_values_edges = run_social_utility_vs_edge_servers()
    
    # Plot and print results
    plot_results(results_clients, x_values_clients, results_edges, x_values_edges)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nExperiment 1 completed in {execution_time:.2f} seconds")
    print("Results saved to 'exp1_social_utility_results.png'")

if __name__ == "__main__":
    main()

