# GACE: Game-theoretic Design of Hierarchical Incentive Mechanism for Federated Learning

This directory contains the implementation of GACE (Game-theoretic based hierArchical inCentive mEchanism) for Hierarchical Federated Learning (HFL) in the FedML framework.

## Overview

GACE consists of three core components:

1. **Client Coalition Rule (φ)**: Forms trust-based coalitions using social network trust relationships
2. **Cluster-Edge Matching Rule (ϑ)**: Uses Kuhn-Munkres algorithm for optimal cluster-edge server matching
3. **Reward Allocation Rule (ζ)**: Three-layer Stackelberg game for incentive design

## Files Structure

### Core Algorithm Files
- `gace_algorithm.py`: Main GACE algorithm implementation
- `gace_torch_hierarchicalfl_step_by_step_exp.py`: GACE experiment runner
- `gace_ra_torch_hierarchicalfl_step_by_step_exp.py`: Random Association (RA) baseline
- `gace_no_torch_hierarchicalfl_step_by_step_exp.py`: GACE-NO (without cluster-edge matching)

### Configuration Files
- `selected_mnist.yaml`: MNIST dataset configuration
- `selected_cifar10.yaml`: CIFAR-10 dataset configuration
- `selected_femnist.yaml`: FEMNIST dataset configuration
- `selected_svhn.yaml`: SVHN dataset configuration

### Experiment Files
- `exp1.py`: Social utility vs number of clients/edge servers (synthetic data)
- `exp2.py`: Cloud server utility vs number of clients/edge servers (synthetic data)
- `exp3mnist.py`: MNIST dataset accuracy and loss experiments
- `exp3cifar10.py`: CIFAR-10 dataset accuracy and loss experiments
- `exp3femnist.py`: FEMNIST dataset accuracy and loss experiments
- `exp3svhn.py`: SVHN dataset accuracy and loss experiments
- `exp4mnist.py`: MNIST dataset with varying low-quality client ratios
- `exp4cifar10.py`: CIFAR-10 dataset with varying low-quality client ratios
- `exp4femnist.py`: FEMNIST dataset with varying low-quality client ratios
- `exp4svhn.py`: SVHN dataset with varying low-quality client ratios

## Installation

1. Install FedML:
```bash
pip install fedml
```

2. Install additional dependencies:
```bash
pip install numpy scipy matplotlib
```

## Usage

### Running Individual Experiments

#### 1. Social Utility Experiments (Synthetic Data)
```bash
# Experiment 1: Social utility vs clients/edge servers
python exp1.py

# Experiment 2: Cloud server utility vs clients/edge servers  
python exp2.py
```

#### 2. Real Dataset Experiments
```bash
# MNIST experiments
python exp3mnist.py
python exp4mnist.py

# CIFAR-10 experiments
python exp3cifar10.py
python exp4cifar10.py

# FEMNIST experiments
python exp3femnist.py
python exp4femnist.py

# SVHN experiments
python exp3svhn.py
python exp4svhn.py
```

#### 3. Individual Algorithm Testing
```bash
# Test GACE algorithm
python gace_torch_hierarchicalfl_step_by_step_exp.py --cf selected_mnist.yaml

# Test RA baseline
python gace_ra_torch_hierarchicalfl_step_by_step_exp.py --cf selected_mnist.yaml

# Test GACE-NO
python gace_no_torch_hierarchicalfl_step_by_step_exp.py --cf selected_mnist.yaml
```

### Configuration Parameters

Key parameters in the YAML configuration files:

- `client_num_in_total`: Total number of clients
- `group_num`: Number of edge servers
- `comm_round`: Number of communication rounds
- `epochs`: Local training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `low_quality_ratio`: Ratio of low-quality clients (0.0-0.5)
- `group_comm_round`: Edge server aggregation frequency

## Algorithm Comparison

The implementation includes the following algorithms for comparison:

1. **GACE**: Complete implementation with all three rules
2. **RA (Random Association)**: Random client-edge server assignment
3. **GACE-NO**: GACE without cluster-edge matching optimization
4. **QAIM**: Quality-Aware Incentive Mechanism (placeholder)
5. **MaxQ**: Maximum Quality mechanism (placeholder)

## Experimental Results

### Synthetic Dataset Results
- **Social Utility**: GACE > GACE-NO > RA
- **Cloud Server Utility**: GACE > GACE-NO > RA
- Performance gap increases with more low-quality clients

### Real Dataset Results
- **Prediction Accuracy**: GACE consistently outperforms baselines
- **Training Loss**: GACE achieves faster convergence
- **Robustness**: GACE maintains performance with increasing low-quality client ratios

## Key Features

### Trust-Based Coalition Formation
- Uses social network trust relationships
- Forms stable coalitions through iterative switching
- Resists malicious client attacks

### Optimal Cluster-Edge Matching
- Kuhn-Munkres algorithm for minimum cost matching
- Minimizes transmission costs
- Improves overall system efficiency

### Three-Layer Stackelberg Game
- Cloud server sets service pricing
- Edge servers determine rewards
- Clients choose data contribution levels
- Proven unique equilibrium solution

## Parameters

### System Parameters
- `M`: Number of edge servers (default: 5)
- `N`: Number of clients (default: 20)
- `alpha`: Trust weight parameter (default: 0.5)

### Cost Parameters
- `Cn_range`: Client training cost range (default: 0.01-0.1)
- `Km_range`: Edge server coordination cost range (default: 0.0-0.001)

### Game Parameters
- `a`: System parameter for edge server utility (default: 2.3)
- `lambda_param`: Weighting parameter for cloud server (default: 4.0)
- `delta_range`: Risk aversion parameter range (default: 1.0-3.0)
- `theta_range`: Reward scaling coefficient range (default: 1.0-2.0)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `client_num_in_total` or `comm_round`
3. **Convergence Issues**: Adjust `convergence_threshold` or `max_iterations`

### Performance Tips

1. Use GPU acceleration when available
2. Adjust batch size based on available memory
3. Reduce number of runs for faster experimentation

## Citation

If you use this implementation, please cite the original GACE paper:

```bibtex
@article{gace2024,
  title={GACE: Game-theoretic Design of Hierarchical Incentive Mechanism for Federated Learning with Dynamic Association},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This implementation follows the same license as the FedML framework.

## Contact

For questions or issues, please refer to the FedML documentation or create an issue in the repository.
