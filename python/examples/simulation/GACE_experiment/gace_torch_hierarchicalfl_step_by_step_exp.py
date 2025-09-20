"""
GACE: Game-theoretic Design of Hierarchical Incentive Mechanism for Federated Learning
Main experiment file for GACE implementation in FedML framework

This file implements the main GACE hierarchical federated learning experiment
with three core components:
1. Client Coalition Rule (φ): Trust-based coalition formation
2. Cluster-Edge Matching Rule (ϑ): Optimal cluster-edge server matching
3. Reward Allocation Rule (ζ): Three-layer Stackelberg game

Author: GACE Implementation
Date: 2024
"""

import fedml
import random
import math
import numpy as np
from scipy.optimize import minimize
from fedml import FedMLRunner
from scipy.sparse import csr_matrix
import copy
import logging
from typing import Dict, List, Tuple, Optional, Union

# Import GACE algorithm
from gace_algorithm import GACEAlgorithm, GACEParameters, GACEResults

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for GACE algorithm
gace_algorithm = None
gace_results = None
trust_matrix = None
transmission_cost = None
training_cost = None
coordination_cost = None
clusters = []
association_matrix = None
reputations = []
low_quality_ratio = 0.0  # Ratio of low-quality clients

def initialize_gace_system(args):
    """Initialize GACE system parameters"""
    global gace_algorithm, trust_matrix, transmission_cost, training_cost, coordination_cost, reputations
    
    # Extract parameters from args
    M = args.group_num  # Number of edge servers
    N = args.client_num_in_total  # Number of clients
    
    # Create GACE parameters
    gace_params = GACEParameters(
        M=M,
        N=N,
        alpha=0.5,  # Trust weight parameter
        Cn_range=(0.01, 0.1),  # Client training cost range
        Km_range=(0.0, 0.001),  # Edge server coordination cost range
        a=2.3,  # System parameter
        lambda_param=4.0,  # Weighting parameter
        delta_range=(1.0, 3.0),  # Risk aversion parameter range
        theta_range=(1.0, 2.0),  # Reward scaling coefficient range
        max_iterations=100,
        convergence_threshold=1e-6
    )
    
    # Initialize GACE algorithm
    gace_algorithm = GACEAlgorithm(gace_params)
    gace_algorithm.initialize_system(seed=args.random_seed)
    
    # Extract system parameters
    trust_matrix = gace_algorithm.trust_matrix
    transmission_cost = gace_algorithm.transmission_cost
    training_cost = gace_algorithm.training_cost
    coordination_cost = gace_algorithm.coordination_cost
    
    # Initialize reputations (higher values for high-quality clients)
    np.random.seed(args.random_seed)
    reputations = np.random.uniform(0.5, 1.0, N)
    
    # Set low-quality clients if specified
    if hasattr(args, 'low_quality_ratio') and args.low_quality_ratio > 0:
        num_low_quality = int(N * args.low_quality_ratio)
        low_quality_indices = np.random.choice(N, num_low_quality, replace=False)
        reputations[low_quality_indices] = np.random.uniform(0.1, 0.3, num_low_quality)
    
    logger.info(f"Initialized GACE system with {M} edge servers and {N} clients")
    logger.info(f"Low quality ratio: {args.low_quality_ratio if hasattr(args, 'low_quality_ratio') else 0.0}")

def execute_gace_algorithm():
    """Execute the complete GACE algorithm"""
    global gace_algorithm, gace_results, clusters, association_matrix
    
    if gace_algorithm is None:
        raise ValueError("GACE system not initialized. Call initialize_gace_system() first.")
    
    logger.info("Executing GACE algorithm...")
    
    # Execute GACE algorithm
    gace_results = gace_algorithm.execute()
    
    # Extract results
    clusters = gace_results.Pi_star
    association_matrix = gace_results.S_star
    
    logger.info(f"GACE algorithm completed. Social utility: {gace_results.U_social:.4f}")
    logger.info(f"Coalition partition: {clusters}")
    
    return gace_results

def get_client_clusters():
    """Get client cluster assignments from GACE algorithm"""
    global clusters, association_matrix
    
    if clusters is None or association_matrix is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    # Convert association matrix to cluster assignments
    client_clusters = {}
    for client_id in range(association_matrix.shape[0]):
        for edge_id in range(association_matrix.shape[1]):
            if association_matrix[client_id, edge_id] == 1:
                client_clusters[client_id] = edge_id
                break
    
    return client_clusters

def calculate_client_utilities():
    """Calculate client utilities based on GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.U_client

def calculate_edge_utilities():
    """Calculate edge server utilities based on GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.U_edge

def calculate_cloud_utility():
    """Calculate cloud server utility based on GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.U_cloud

def calculate_social_utility():
    """Calculate social utility based on GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.U_social

def get_optimal_pricing():
    """Get optimal service pricing from GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.P_star

def get_optimal_rewards():
    """Get optimal edge server rewards from GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.Gamma_star

def get_optimal_data_plans():
    """Get optimal client data plans from GACE results"""
    global gace_results
    
    if gace_results is None:
        raise ValueError("GACE algorithm not executed. Call execute_gace_algorithm() first.")
    
    return gace_results.D_star

def get_trust_matrix():
    """Get trust matrix between clients"""
    global trust_matrix
    return trust_matrix

def get_transmission_costs():
    """Get transmission cost matrix"""
    global transmission_cost
    return transmission_cost

def get_training_costs():
    """Get training cost vector for clients"""
    global training_cost
    return training_cost

def get_coordination_costs():
    """Get coordination cost vector for edge servers"""
    global coordination_cost
    return coordination_cost

def get_reputations():
    """Get reputation vector for clients"""
    global reputations
    return reputations

def update_reputations(client_id, performance_score):
    """Update client reputation based on performance"""
    global reputations
    
    if reputations is None:
        return
    
    # Update reputation using exponential moving average
    alpha = 0.1  # Learning rate
    reputations[client_id] = (1 - alpha) * reputations[client_id] + alpha * performance_score
    
    # Ensure reputation stays within bounds
    reputations[client_id] = max(0.0, min(1.0, reputations[client_id]))

def get_client_association_matrix():
    """Get client-edge server association matrix"""
    global association_matrix
    return association_matrix

def get_cluster_partition():
    """Get cluster partition from GACE algorithm"""
    global clusters
    return clusters

def print_gace_results():
    """Print GACE algorithm results"""
    global gace_results
    
    if gace_results is None:
        logger.warning("GACE algorithm not executed. No results to print.")
        return
    
    print("\n" + "="*60)
    print("GACE ALGORITHM RESULTS")
    print("="*60)
    print(f"Social Utility: {gace_results.U_social:.4f}")
    print(f"Cloud Server Utility: {gace_results.U_cloud:.4f}")
    print(f"Average Edge Server Utility: {np.mean(gace_results.U_edge):.4f}")
    print(f"Average Client Utility: {np.mean(gace_results.U_client):.4f}")
    print(f"Optimal Service Price: {gace_results.P_star:.4f}")
    print(f"Optimal Edge Server Rewards: {gace_results.Gamma_star}")
    print(f"Optimal Client Data Plans: {gace_results.D_star}")
    
    print(f"\nCoalition Partition:")
    for i, cluster in enumerate(gace_results.Pi_star):
        if cluster:
            print(f"  Cluster {i}: {cluster}")
    
    print("="*60)

def reset_gace_system():
    """Reset GACE system for new experiment"""
    global gace_algorithm, gace_results, clusters, association_matrix
    
    gace_algorithm = None
    gace_results = None
    clusters = []
    association_matrix = None
    
    logger.info("GACE system reset")

# Main function for testing
def main():
    """Main function for testing GACE algorithm"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=str, default='selected_mnist.yaml', help='config file')
    parser.add_argument('--group_num', type=int, default=5, help='number of edge servers')
    parser.add_argument('--client_num_in_total', type=int, default=20, help='number of clients')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--low_quality_ratio', type=float, default=0.0, help='ratio of low-quality clients')
    
    args = parser.parse_args()
    
    # Initialize GACE system
    initialize_gace_system(args)
    
    # Execute GACE algorithm
    results = execute_gace_algorithm()
    
    # Print results
    print_gace_results()
    
    # Get cluster assignments
    client_clusters = get_client_clusters()
    print(f"\nClient-Edge Server Assignments: {client_clusters}")

if __name__ == "__main__":
    main()

