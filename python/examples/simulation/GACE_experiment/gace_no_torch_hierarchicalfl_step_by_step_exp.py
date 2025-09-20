"""
GACE-NO: GACE without Cluster-Edge Matching optimization
This file implements GACE-NO where client coalition formation is used but cluster-edge matching is skipped

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

# Global variables for GACE-NO algorithm
gace_no_algorithm = None
gace_no_results = None
trust_matrix = None
transmission_cost = None
training_cost = None
coordination_cost = None
clusters = []
association_matrix = None
reputations = []
low_quality_ratio = 0.0  # Ratio of low-quality clients

def initialize_gace_no_system(args):
    """Initialize GACE-NO system parameters"""
    global gace_no_algorithm, trust_matrix, transmission_cost, training_cost, coordination_cost, reputations
    
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
    gace_no_algorithm = GACEAlgorithm(gace_params)
    gace_no_algorithm.initialize_system(seed=args.random_seed)
    
    # Extract system parameters
    trust_matrix = gace_no_algorithm.trust_matrix
    transmission_cost = gace_no_algorithm.transmission_cost
    training_cost = gace_no_algorithm.training_cost
    coordination_cost = gace_no_algorithm.coordination_cost
    
    # Initialize reputations (higher values for high-quality clients)
    np.random.seed(args.random_seed)
    reputations = np.random.uniform(0.5, 1.0, N)
    
    # Set low-quality clients if specified
    if hasattr(args, 'low_quality_ratio') and args.low_quality_ratio > 0:
        num_low_quality = int(N * args.low_quality_ratio)
        low_quality_indices = np.random.choice(N, num_low_quality, replace=False)
        reputations[low_quality_indices] = np.random.uniform(0.1, 0.3, num_low_quality)
    
    logger.info(f"Initialized GACE-NO system with {M} edge servers and {N} clients")
    logger.info(f"Low quality ratio: {args.low_quality_ratio if hasattr(args, 'low_quality_ratio') else 0.0}")

def execute_gace_no_algorithm():
    """Execute GACE-NO algorithm (coalition formation + reward allocation, no cluster-edge matching)"""
    global gace_no_algorithm, gace_no_results, clusters, association_matrix
    
    if gace_no_algorithm is None:
        raise ValueError("GACE-NO system not initialized. Call initialize_gace_no_system() first.")
    
    logger.info("Executing GACE-NO algorithm (without cluster-edge matching)...")
    
    # Step 1: Client Coalition Rule (same as GACE)
    clusters = gace_no_algorithm.client_coalition_rule()
    
    # Step 2: Skip Cluster-Edge Matching - use initial association matrix
    association_matrix = _initialize_association_matrix_no_optimization(clusters)
    
    # Step 3: Reward Allocation Rule (same as GACE)
    gace_no_results = _execute_reward_allocation_gace_no()
    
    logger.info(f"GACE-NO algorithm completed. Social utility: {gace_no_results.U_social:.4f}")
    logger.info(f"Coalition partition: {clusters}")
    
    return gace_no_results

def _initialize_association_matrix_no_optimization(partition):
    """Initialize association matrix without optimization (direct assignment)"""
    global gace_no_algorithm
    
    M = gace_no_algorithm.params.M
    N = gace_no_algorithm.params.N
    
    # Create association matrix based on coalition partition (no optimization)
    S = np.zeros((N, M))
    for cluster_idx, cluster in enumerate(partition):
        for client_n in cluster:
            # Direct assignment: cluster i -> edge server i
            S[client_n, cluster_idx] = 1
    
    return S

def _execute_reward_allocation_gace_no():
    """Execute reward allocation rule for GACE-NO (same as GACE)"""
    global gace_no_algorithm, clusters, association_matrix
    
    # Get clusters from association matrix
    clusters_from_association = _get_clusters_from_association(association_matrix)
    
    # Initialize strategies
    P = 1.0  # Initial unit service price
    Gamma = np.ones(gace_no_algorithm.params.M)  # Initial edge server rewards
    D = np.zeros(gace_no_algorithm.params.N)  # Initial client data plans
    
    iteration = 0
    converged = False
    
    while not converged and iteration < gace_no_algorithm.params.max_iterations:
        P_prev = P
        Gamma_prev = copy.deepcopy(Gamma)
        D_prev = copy.deepcopy(D)
        
        # Step 1: Solve Data Plan (DP) game
        D = _solve_data_plan_game(Gamma, clusters_from_association)
        
        # Step 2: Solve Reward Declaration (RD) game
        Gamma = _solve_reward_declaration_game(P, D, clusters_from_association)
        
        # Step 3: Solve Service Pricing (SP) game
        P = _solve_service_pricing_game(Gamma, D, clusters_from_association)
        
        # Check convergence
        if (abs(P - P_prev) < gace_no_algorithm.params.convergence_threshold and
            np.allclose(Gamma, Gamma_prev, atol=gace_no_algorithm.params.convergence_threshold) and
            np.allclose(D, D_prev, atol=gace_no_algorithm.params.convergence_threshold)):
            converged = True
        
        iteration += 1
    
    logger.info(f"GACE-NO reward allocation converged in {iteration} iterations")
    
    # Calculate utilities
    utilities = _calculate_utilities_gace_no(P, Gamma, D, association_matrix)
    
    # Create results
    results = GACEResults(
        Pi_star=clusters_from_association,
        S_star=association_matrix,
        P_star=P,
        Gamma_star=Gamma,
        D_star=D,
        U_cloud=utilities['cloud'],
        U_edge=utilities['edge'],
        U_client=utilities['client'],
        U_social=utilities['social'],
        iterations=iteration,
        convergence_info={}
    )
    
    return results

def _get_clusters_from_association(S):
    """Get cluster assignments from association matrix"""
    M = S.shape[1]
    clusters = [[] for _ in range(M)]
    for n in range(S.shape[0]):
        for m in range(M):
            if S[n, m] == 1:
                clusters[m].append(n)
                break
    return clusters

def _solve_data_plan_game(Gamma, clusters):
    """Solve Data Plan (DP) game"""
    global gace_no_algorithm
    
    D = np.zeros(gace_no_algorithm.params.N)
    
    for m in range(gace_no_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) <= 1:
            if len(cluster) == 1:
                D[cluster[0]] = 1.0
            continue
        
        for n in cluster:
            if n in cluster:
                total_cost = np.sum([gace_no_algorithm.training_cost[i] for i in cluster])
                if total_cost > 0:
                    D[n] = ((len(cluster) - 1) * Gamma[m] / (total_cost ** 2)) * \
                           (total_cost - (len(cluster) - 1) * gace_no_algorithm.training_cost[n])
                    D[n] = max(1.0, D[n])
                else:
                    D[n] = 1.0
    
    return D

def _solve_reward_declaration_game(P, D, clusters):
    """Solve Reward Declaration (RD) game"""
    global gace_no_algorithm
    
    Gamma = np.zeros(gace_no_algorithm.params.M)
    
    for m in range(gace_no_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) == 0:
            Gamma[m] = 1.0
            continue
        
        total_cost = np.sum([gace_no_algorithm.training_cost[i] for i in cluster])
        if total_cost > 0:
            B_m = (len(cluster) - 1) / total_cost
            
            theta_m = np.random.uniform(gace_no_algorithm.params.theta_range[0], gace_no_algorithm.params.theta_range[1])
            delta_m = np.random.uniform(gace_no_algorithm.params.delta_range[0], gace_no_algorithm.params.delta_range[1])
            
            if P * B_m > 0:
                Gamma[m] = theta_m / np.log(gace_no_algorithm.params.a) - delta_m / (P * B_m)
                Gamma[m] = max(2.0, Gamma[m])
            else:
                Gamma[m] = 2.0
        else:
            Gamma[m] = 2.0
    
    return Gamma

def _solve_service_pricing_game(Gamma, D, clusters):
    """Solve Service Pricing (SP) game"""
    global gace_no_algorithm
    
    B_values = []
    delta_values = []
    theta_values = []
    
    for m in range(gace_no_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) == 0:
            B_values.append(1.0)
            delta_values.append(1.0)
            theta_values.append(2.0)
            continue
        
        total_cost = np.sum([gace_no_algorithm.training_cost[i] for i in cluster])
        if total_cost > 0:
            B_m = (len(cluster) - 1) / total_cost
            B_values.append(B_m)
            delta_values.append(np.random.uniform(gace_no_algorithm.params.delta_range[0], gace_no_algorithm.params.delta_range[1]))
            theta_values.append(np.random.uniform(gace_no_algorithm.params.theta_range[0], gace_no_algorithm.params.theta_range[1]))
        else:
            B_values.append(1.0)
            delta_values.append(1.0)
            theta_values.append(2.0)
    
    sum_delta = np.sum(delta_values)
    sum_theta_B = np.sum([theta_values[i] * B_values[i] for i in range(len(B_values))])
    
    if sum_theta_B > 0:
        discriminant = (sum_delta ** 2 + 4 * gace_no_algorithm.params.lambda_param * sum_delta + 
                       4 * gace_no_algorithm.params.lambda_param * np.log(gace_no_algorithm.params.a) * sum_delta / sum_theta_B)
        P = (sum_delta + np.sqrt(discriminant)) / (2 * sum_theta_B / np.log(gace_no_algorithm.params.a) + 2)
    else:
        P = 2.0
    
    return max(1.0, P)

def _calculate_utilities_gace_no(P, Gamma, D, S):
    """Calculate utilities for all participants in GACE-NO"""
    global gace_no_algorithm
    
    clusters = _get_clusters_from_association(S)
    
    # Cloud server utility
    total_data = np.sum(D)
    U_cloud = (gace_no_algorithm.params.lambda_param * np.log(total_data + 1) + 2.0 + total_data * 0.2 + 
              gace_no_algorithm.params.N * 0.025 + gace_no_algorithm.params.M * 0.1 - P * total_data * 0.005) / 4.0
    
    # Edge server utilities
    U_edge = np.zeros(gace_no_algorithm.params.M)
    for m in range(gace_no_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) > 0:
            cluster_data = np.sum([D[i] for i in cluster])
            U_edge[m] = (np.log(gace_no_algorithm.params.a) * (P * cluster_data + 1) + 0.75 + 
                        cluster_data * 0.125 + len(cluster) * 0.05 + gace_no_algorithm.params.N * 0.0125 + 
                        gace_no_algorithm.params.M * 0.05 - Gamma[m] * 0.005 - 
                        gace_no_algorithm.coordination_cost[m] * len(cluster) * 0.00125) / 4.0
        else:
            U_edge[m] = (0.25 + gace_no_algorithm.params.N * 0.0125 + gace_no_algorithm.params.M * 0.05) / 4.0
    
    # Client utilities
    U_client = np.zeros(gace_no_algorithm.params.N)
    for n in range(gace_no_algorithm.params.N):
        for m in range(gace_no_algorithm.params.M):
            if S[n, m] == 1:
                cluster = clusters[m]
                if len(cluster) > 0:
                    cluster_data = np.sum([D[i] for i in cluster])
                    if cluster_data > 0:
                        U_client[n] = (D[n] / cluster_data * Gamma[m] + 0.375 + 
                                     D[n] * 0.075 + len(cluster) * 0.025 + gace_no_algorithm.params.M * 0.0125 - 
                                     D[n] * gace_no_algorithm.training_cost[n] * 0.00125 - 
                                     gace_no_algorithm.transmission_cost[m, n] * 0.00125) / 4.0
                    else:
                        U_client[n] = (0.25 + gace_no_algorithm.params.M * 0.0125) / 4.0
                else:
                    U_client[n] = (0.25 + gace_no_algorithm.params.M * 0.0125) / 4.0
                break
    
    # Social utility
    U_social = U_cloud + np.sum(U_edge) + np.sum(U_client)
    
    return {
        'cloud': U_cloud,
        'edge': U_edge,
        'client': U_client,
        'social': U_social
    }

def get_client_clusters():
    """Get client cluster assignments from GACE-NO algorithm"""
    global clusters, association_matrix
    
    if clusters is None or association_matrix is None:
        raise ValueError("GACE-NO algorithm not executed. Call execute_gace_no_algorithm() first.")
    
    # Convert association matrix to cluster assignments
    client_clusters = {}
    for client_id in range(association_matrix.shape[0]):
        for edge_id in range(association_matrix.shape[1]):
            if association_matrix[client_id, edge_id] == 1:
                client_clusters[client_id] = edge_id
                break
    
    return client_clusters

def calculate_client_utilities():
    """Calculate client utilities based on GACE-NO results"""
    global gace_no_results
    
    if gace_no_results is None:
        raise ValueError("GACE-NO algorithm not executed. Call execute_gace_no_algorithm() first.")
    
    return gace_no_results.U_client

def calculate_edge_utilities():
    """Calculate edge server utilities based on GACE-NO results"""
    global gace_no_results
    
    if gace_no_results is None:
        raise ValueError("GACE-NO algorithm not executed. Call execute_gace_no_algorithm() first.")
    
    return gace_no_results.U_edge

def calculate_cloud_utility():
    """Calculate cloud server utility based on GACE-NO results"""
    global gace_no_results
    
    if gace_no_results is None:
        raise ValueError("GACE-NO algorithm not executed. Call execute_gace_no_algorithm() first.")
    
    return gace_no_results.U_cloud

def calculate_social_utility():
    """Calculate social utility based on GACE-NO results"""
    global gace_no_results
    
    if gace_no_results is None:
        raise ValueError("GACE-NO algorithm not executed. Call execute_gace_no_algorithm() first.")
    
    return gace_no_results.U_social

def print_gace_no_results():
    """Print GACE-NO algorithm results"""
    global gace_no_results
    
    if gace_no_results is None:
        logger.warning("GACE-NO algorithm not executed. No results to print.")
        return
    
    print("\n" + "="*60)
    print("GACE-NO ALGORITHM RESULTS")
    print("="*60)
    print(f"Social Utility: {gace_no_results.U_social:.4f}")
    print(f"Cloud Server Utility: {gace_no_results.U_cloud:.4f}")
    print(f"Average Edge Server Utility: {np.mean(gace_no_results.U_edge):.4f}")
    print(f"Average Client Utility: {np.mean(gace_no_results.U_client):.4f}")
    print(f"Optimal Service Price: {gace_no_results.P_star:.4f}")
    print(f"Optimal Edge Server Rewards: {gace_no_results.Gamma_star}")
    print(f"Optimal Client Data Plans: {gace_no_results.D_star}")
    
    print(f"\nCoalition Partition (No Optimization):")
    for i, cluster in enumerate(gace_no_results.Pi_star):
        if cluster:
            print(f"  Cluster {i}: {cluster}")
    
    print("="*60)

def reset_gace_system():
    """Reset GACE-NO system for new experiment"""
    global gace_no_algorithm, gace_no_results, clusters, association_matrix
    
    gace_no_algorithm = None
    gace_no_results = None
    clusters = []
    association_matrix = None
    
    logger.info("GACE-NO system reset")

# Main function for testing
def main():
    """Main function for testing GACE-NO algorithm"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=str, default='selected_mnist.yaml', help='config file')
    parser.add_argument('--group_num', type=int, default=5, help='number of edge servers')
    parser.add_argument('--client_num_in_total', type=int, default=20, help='number of clients')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--low_quality_ratio', type=float, default=0.0, help='ratio of low-quality clients')
    
    args = parser.parse_args()
    
    # Initialize GACE-NO system
    initialize_gace_no_system(args)
    
    # Execute GACE-NO algorithm
    results = execute_gace_no_algorithm()
    
    # Print results
    print_gace_no_results()
    
    # Get cluster assignments
    client_clusters = get_client_clusters()
    print(f"\nClient-Edge Server Assignments: {client_clusters}")

if __name__ == "__main__":
    main()

