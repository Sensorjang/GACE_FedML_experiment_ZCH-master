"""
GACE-RA: Random Association baseline for GACE comparison
This file implements the Random Association (RA) baseline where clients are randomly assigned to edge servers

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

# Import GACE algorithm for comparison
from gace_algorithm import GACEAlgorithm, GACEParameters, GACEResults

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for RA algorithm
ra_algorithm = None
ra_results = None
trust_matrix = None
transmission_cost = None
training_cost = None
coordination_cost = None
clusters = []
association_matrix = None
reputations = []
low_quality_ratio = 0.0  # Ratio of low-quality clients

def initialize_ra_system(args):
    """Initialize RA system parameters"""
    global ra_algorithm, trust_matrix, transmission_cost, training_cost, coordination_cost, reputations
    
    # Extract parameters from args
    M = args.group_num  # Number of edge servers
    N = args.client_num_in_total  # Number of clients
    
    # Create GACE parameters (we'll use GACE framework but with random assignment)
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
    ra_algorithm = GACEAlgorithm(gace_params)
    ra_algorithm.initialize_system(seed=args.random_seed)
    
    # Extract system parameters
    trust_matrix = ra_algorithm.trust_matrix
    transmission_cost = ra_algorithm.transmission_cost
    training_cost = ra_algorithm.training_cost
    coordination_cost = ra_algorithm.coordination_cost
    
    # Initialize reputations (higher values for high-quality clients)
    np.random.seed(args.random_seed)
    reputations = np.random.uniform(0.5, 1.0, N)
    
    # Set low-quality clients if specified
    if hasattr(args, 'low_quality_ratio') and args.low_quality_ratio > 0:
        num_low_quality = int(N * args.low_quality_ratio)
        low_quality_indices = np.random.choice(N, num_low_quality, replace=False)
        reputations[low_quality_indices] = np.random.uniform(0.1, 0.3, num_low_quality)
    
    logger.info(f"Initialized RA system with {M} edge servers and {N} clients")
    logger.info(f"Low quality ratio: {args.low_quality_ratio if hasattr(args, 'low_quality_ratio') else 0.0}")

def execute_ra_algorithm():
    """Execute RA algorithm with random client-edge server assignment"""
    global ra_algorithm, ra_results, clusters, association_matrix
    
    if ra_algorithm is None:
        raise ValueError("RA system not initialized. Call initialize_ra_system() first.")
    
    logger.info("Executing RA algorithm with random assignment...")
    
    # Step 1: Random client coalition formation (no trust-based optimization)
    clusters = _random_coalition_formation()
    
    # Step 2: Random cluster-edge matching (no optimization)
    association_matrix = _random_cluster_edge_matching()
    
    # Step 3: Execute reward allocation rule (same as GACE)
    ra_results = _execute_reward_allocation_ra()
    
    logger.info(f"RA algorithm completed. Social utility: {ra_results.U_social:.4f}")
    logger.info(f"Random coalition partition: {clusters}")
    
    return ra_results

def _random_coalition_formation():
    """Random coalition formation without trust-based optimization"""
    global ra_algorithm
    
    M = ra_algorithm.params.M
    N = ra_algorithm.params.N
    
    # Randomly assign clients to clusters
    clusters = [[] for _ in range(M)]
    for client_id in range(N):
        cluster_id = np.random.randint(0, M)
        clusters[cluster_id].append(client_id)
    
    return clusters

def _random_cluster_edge_matching():
    """Random cluster-edge server matching without optimization"""
    global ra_algorithm, clusters
    
    M = ra_algorithm.params.M
    N = ra_algorithm.params.N
    
    # Create random association matrix
    association_matrix = np.zeros((N, M))
    for client_id in range(N):
        # Find which cluster the client belongs to
        for cluster_id, cluster in enumerate(clusters):
            if client_id in cluster:
                # Randomly assign to an edge server
                edge_id = np.random.randint(0, M)
                association_matrix[client_id, edge_id] = 1
                break
    
    return association_matrix

def _execute_reward_allocation_ra():
    """Execute reward allocation rule for RA (same as GACE)"""
    global ra_algorithm, clusters, association_matrix
    
    # Get clusters from association matrix
    clusters_from_association = _get_clusters_from_association(association_matrix)
    
    # Initialize strategies
    P = 1.0  # Initial unit service price
    Gamma = np.ones(ra_algorithm.params.M)  # Initial edge server rewards
    D = np.zeros(ra_algorithm.params.N)  # Initial client data plans
    
    iteration = 0
    converged = False
    
    while not converged and iteration < ra_algorithm.params.max_iterations:
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
        if (abs(P - P_prev) < ra_algorithm.params.convergence_threshold and
            np.allclose(Gamma, Gamma_prev, atol=ra_algorithm.params.convergence_threshold) and
            np.allclose(D, D_prev, atol=ra_algorithm.params.convergence_threshold)):
            converged = True
        
        iteration += 1
    
    logger.info(f"RA reward allocation converged in {iteration} iterations")
    
    # Calculate utilities
    utilities = _calculate_utilities_ra(P, Gamma, D, association_matrix)
    
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
    global ra_algorithm
    
    D = np.zeros(ra_algorithm.params.N)
    
    for m in range(ra_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) <= 1:
            if len(cluster) == 1:
                D[cluster[0]] = 1.0
            continue
        
        for n in cluster:
            if n in cluster:
                total_cost = np.sum([ra_algorithm.training_cost[i] for i in cluster])
                if total_cost > 0:
                    D[n] = ((len(cluster) - 1) * Gamma[m] / (total_cost ** 2)) * \
                           (total_cost - (len(cluster) - 1) * ra_algorithm.training_cost[n])
                    D[n] = max(1.0, D[n])
                else:
                    D[n] = 1.0
    
    return D

def _solve_reward_declaration_game(P, D, clusters):
    """Solve Reward Declaration (RD) game"""
    global ra_algorithm
    
    Gamma = np.zeros(ra_algorithm.params.M)
    
    for m in range(ra_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) == 0:
            Gamma[m] = 1.0
            continue
        
        total_cost = np.sum([ra_algorithm.training_cost[i] for i in cluster])
        if total_cost > 0:
            B_m = (len(cluster) - 1) / total_cost
            
            theta_m = np.random.uniform(ra_algorithm.params.theta_range[0], ra_algorithm.params.theta_range[1])
            delta_m = np.random.uniform(ra_algorithm.params.delta_range[0], ra_algorithm.params.delta_range[1])
            
            if P * B_m > 0:
                Gamma[m] = theta_m / np.log(ra_algorithm.params.a) - delta_m / (P * B_m)
                Gamma[m] = max(2.0, Gamma[m])
            else:
                Gamma[m] = 2.0
        else:
            Gamma[m] = 2.0
    
    return Gamma

def _solve_service_pricing_game(Gamma, D, clusters):
    """Solve Service Pricing (SP) game"""
    global ra_algorithm
    
    B_values = []
    delta_values = []
    theta_values = []
    
    for m in range(ra_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) == 0:
            B_values.append(1.0)
            delta_values.append(1.0)
            theta_values.append(2.0)
            continue
        
        total_cost = np.sum([ra_algorithm.training_cost[i] for i in cluster])
        if total_cost > 0:
            B_m = (len(cluster) - 1) / total_cost
            B_values.append(B_m)
            delta_values.append(np.random.uniform(ra_algorithm.params.delta_range[0], ra_algorithm.params.delta_range[1]))
            theta_values.append(np.random.uniform(ra_algorithm.params.theta_range[0], ra_algorithm.params.theta_range[1]))
        else:
            B_values.append(1.0)
            delta_values.append(1.0)
            theta_values.append(2.0)
    
    sum_delta = np.sum(delta_values)
    sum_theta_B = np.sum([theta_values[i] * B_values[i] for i in range(len(B_values))])
    
    if sum_theta_B > 0:
        discriminant = (sum_delta ** 2 + 4 * ra_algorithm.params.lambda_param * sum_delta + 
                       4 * ra_algorithm.params.lambda_param * np.log(ra_algorithm.params.a) * sum_delta / sum_theta_B)
        P = (sum_delta + np.sqrt(discriminant)) / (2 * sum_theta_B / np.log(ra_algorithm.params.a) + 2)
    else:
        P = 2.0
    
    return max(1.0, P)

def _calculate_utilities_ra(P, Gamma, D, S):
    """Calculate utilities for all participants in RA"""
    global ra_algorithm
    
    clusters = _get_clusters_from_association(S)
    
    # Cloud server utility
    total_data = np.sum(D)
    U_cloud = (ra_algorithm.params.lambda_param * np.log(total_data + 1) + 2.0 + total_data * 0.2 + 
              ra_algorithm.params.N * 0.025 + ra_algorithm.params.M * 0.1 - P * total_data * 0.005) / 4.0
    
    # Edge server utilities
    U_edge = np.zeros(ra_algorithm.params.M)
    for m in range(ra_algorithm.params.M):
        cluster = clusters[m]
        if len(cluster) > 0:
            cluster_data = np.sum([D[i] for i in cluster])
            U_edge[m] = (np.log(ra_algorithm.params.a) * (P * cluster_data + 1) + 0.75 + 
                        cluster_data * 0.125 + len(cluster) * 0.05 + ra_algorithm.params.N * 0.0125 + 
                        ra_algorithm.params.M * 0.05 - Gamma[m] * 0.005 - 
                        ra_algorithm.coordination_cost[m] * len(cluster) * 0.00125) / 4.0
        else:
            U_edge[m] = (0.25 + ra_algorithm.params.N * 0.0125 + ra_algorithm.params.M * 0.05) / 4.0
    
    # Client utilities
    U_client = np.zeros(ra_algorithm.params.N)
    for n in range(ra_algorithm.params.N):
        for m in range(ra_algorithm.params.M):
            if S[n, m] == 1:
                cluster = clusters[m]
                if len(cluster) > 0:
                    cluster_data = np.sum([D[i] for i in cluster])
                    if cluster_data > 0:
                        U_client[n] = (D[n] / cluster_data * Gamma[m] + 0.375 + 
                                     D[n] * 0.075 + len(cluster) * 0.025 + ra_algorithm.params.M * 0.0125 - 
                                     D[n] * ra_algorithm.training_cost[n] * 0.00125 - 
                                     ra_algorithm.transmission_cost[m, n] * 0.00125) / 4.0
                    else:
                        U_client[n] = (0.25 + ra_algorithm.params.M * 0.0125) / 4.0
                else:
                    U_client[n] = (0.25 + ra_algorithm.params.M * 0.0125) / 4.0
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
    """Get client cluster assignments from RA algorithm"""
    global clusters, association_matrix
    
    if clusters is None or association_matrix is None:
        raise ValueError("RA algorithm not executed. Call execute_ra_algorithm() first.")
    
    # Convert association matrix to cluster assignments
    client_clusters = {}
    for client_id in range(association_matrix.shape[0]):
        for edge_id in range(association_matrix.shape[1]):
            if association_matrix[client_id, edge_id] == 1:
                client_clusters[client_id] = edge_id
                break
    
    return client_clusters

def calculate_client_utilities():
    """Calculate client utilities based on RA results"""
    global ra_results
    
    if ra_results is None:
        raise ValueError("RA algorithm not executed. Call execute_ra_algorithm() first.")
    
    return ra_results.U_client

def calculate_edge_utilities():
    """Calculate edge server utilities based on RA results"""
    global ra_results
    
    if ra_results is None:
        raise ValueError("RA algorithm not executed. Call execute_ra_algorithm() first.")
    
    return ra_results.U_edge

def calculate_cloud_utility():
    """Calculate cloud server utility based on RA results"""
    global ra_results
    
    if ra_results is None:
        raise ValueError("RA algorithm not executed. Call execute_ra_algorithm() first.")
    
    return ra_results.U_cloud

def calculate_social_utility():
    """Calculate social utility based on RA results"""
    global ra_results
    
    if ra_results is None:
        raise ValueError("RA algorithm not executed. Call execute_ra_algorithm() first.")
    
    return ra_results.U_social

def print_ra_results():
    """Print RA algorithm results"""
    global ra_results
    
    if ra_results is None:
        logger.warning("RA algorithm not executed. No results to print.")
        return
    
    print("\n" + "="*60)
    print("RA ALGORITHM RESULTS")
    print("="*60)
    print(f"Social Utility: {ra_results.U_social:.4f}")
    print(f"Cloud Server Utility: {ra_results.U_cloud:.4f}")
    print(f"Average Edge Server Utility: {np.mean(ra_results.U_edge):.4f}")
    print(f"Average Client Utility: {np.mean(ra_results.U_client):.4f}")
    print(f"Optimal Service Price: {ra_results.P_star:.4f}")
    print(f"Optimal Edge Server Rewards: {ra_results.Gamma_star}")
    print(f"Optimal Client Data Plans: {ra_results.D_star}")
    
    print(f"\nRandom Coalition Partition:")
    for i, cluster in enumerate(ra_results.Pi_star):
        if cluster:
            print(f"  Cluster {i}: {cluster}")
    
    print("="*60)

# Main function for testing
def main():
    """Main function for testing RA algorithm"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=str, default='selected_mnist.yaml', help='config file')
    parser.add_argument('--group_num', type=int, default=5, help='number of edge servers')
    parser.add_argument('--client_num_in_total', type=int, default=20, help='number of clients')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--low_quality_ratio', type=float, default=0.0, help='ratio of low-quality clients')
    
    args = parser.parse_args()
    
    # Initialize RA system
    initialize_ra_system(args)
    
    # Execute RA algorithm
    results = execute_ra_algorithm()
    
    # Print results
    print_ra_results()
    
    # Get cluster assignments
    client_clusters = get_client_clusters()
    print(f"\nClient-Edge Server Assignments: {client_clusters}")

if __name__ == "__main__":
    main()

