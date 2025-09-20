# GACE FedML Implementation Summary

## 项目概述

本项目成功实现了GACE (Game-theoretic based hierArchical inCentive mEchanism) 在FedML框架中的完整实验系统。GACE是一个基于博弈论的分层联邦学习激励机制，包含三个核心组件：

1. **客户端联盟规则 (φ)**: 基于社交网络信任关系的联盟形成
2. **集群-边缘匹配规则 (ϑ)**: 使用Kuhn-Munkres算法的最优匹配
3. **奖励分配规则 (ζ)**: 三层Stackelberg博弈的激励设计

## 已实现文件列表

### 核心算法文件
- ✅ `gace_algorithm.py` - GACE核心算法实现
- ✅ `gace_torch_hierarchicalfl_step_by_step_exp.py` - GACE实验运行器
- ✅ `gace_ra_torch_hierarchicalfl_step_by_step_exp.py` - 随机关联(RA)基线算法
- ✅ `gace_no_torch_hierarchicalfl_step_by_step_exp.py` - GACE-NO算法(无集群-边缘匹配优化)

### 配置文件
- ✅ `selected_mnist.yaml` - MNIST数据集配置
- ✅ `selected_cifar10.yaml` - CIFAR-10数据集配置
- ✅ `selected_femnist.yaml` - FEMNIST数据集配置
- ✅ `selected_svhn.yaml` - SVHN数据集配置

### 实验文件
- ✅ `exp1.py` - 实验1: 社会效用 vs 客户端/边缘服务器数量 (合成数据)
- ✅ `exp2.py` - 实验2: 云服务器效用 vs 客户端/边缘服务器数量 (合成数据)
- ✅ `exp3mnist.py` - 实验3: MNIST数据集准确度和损失分析
- ✅ `exp3cifar10.py` - 实验3: CIFAR-10数据集准确度和损失分析
- ✅ `exp3femnist.py` - 实验3: FEMNIST数据集准确度和损失分析
- ✅ `exp3svhn.py` - 实验3: SVHN数据集准确度和损失分析
- ✅ `exp4mnist.py` - 实验4: MNIST数据集低质量客户端分析
- ✅ `exp4cifar10.py` - 实验4: CIFAR-10数据集低质量客户端分析
- ✅ `exp4femnist.py` - 实验4: FEMNIST数据集低质量客户端分析
- ✅ `exp4svhn.py` - 实验4: SVHN数据集低质量客户端分析

### 工具文件
- ✅ `run_gace_experiments.py` - 实验运行脚本
- ✅ `test_gace_implementation.py` - 实现测试脚本
- ✅ `README.md` - 详细使用说明
- ✅ `__init__.py` - 包初始化文件

## 算法对比

实现了以下算法进行对比：

1. **GACE**: 完整的GACE实现，包含所有三个规则
2. **RA (Random Association)**: 随机客户端-边缘服务器分配
3. **GACE-NO**: GACE算法但不使用集群-边缘匹配优化
4. **QAIM**: 质量感知激励机制 (占位符实现)
5. **MaxQ**: 最大质量机制 (占位符实现)

## 实验设计

### 实验1 & 2: 合成数据效用分析
- **目标**: 分析社会效用和云服务器效用随客户端和边缘服务器数量的变化
- **对比算法**: GACE, RA, GACE-NO
- **输出**: 效用对比图表

### 实验3: 真实数据集准确度分析
- **目标**: 在真实数据集上比较各算法的预测准确度和训练损失
- **数据集**: MNIST, CIFAR-10, FEMNIST, SVHN
- **对比算法**: GACE, RA, GACE-NO, QAIM, MaxQ
- **输出**: 准确度和损失曲线图

### 实验4: 低质量客户端鲁棒性分析
- **目标**: 分析各算法在不同低质量客户端比例下的性能表现
- **低质量比例**: 10%, 20%, 30%, 40%, 50%
- **对比算法**: GACE, RA, GACE-NO, QAIM, MaxQ
- **输出**: 准确度 vs 低质量比例图表

## 使用方法

### 快速开始
```bash
# 检查依赖和文件结构
python run_gace_experiments.py --check

# 运行所有实验
python run_gace_experiments.py --all

# 运行特定实验
python run_gace_experiments.py --exp1
python run_gace_experiments.py --exp3 --dataset mnist
```

### 单独运行实验
```bash
# 实验1: 社会效用分析
python exp1.py

# 实验2: 云服务器效用分析
python exp2.py

# 实验3: 真实数据集实验
python exp3mnist.py --cf selected_mnist.yaml
python exp3cifar10.py --cf selected_cifar10.yaml
python exp3femnist.py --cf selected_femnist.yaml
python exp3svhn.py --cf selected_svhn.yaml

# 实验4: 低质量客户端分析
python exp4mnist.py --cf selected_mnist.yaml
python exp4cifar10.py --cf selected_cifar10.yaml
python exp4femnist.py --cf selected_femnist.yaml
python exp4svhn.py --cf selected_svhn.yaml
```

### 测试单个算法
```bash
# 测试GACE算法
python gace_torch_hierarchicalfl_step_by_step_exp.py --cf selected_mnist.yaml

# 测试RA算法
python gace_ra_torch_hierarchicalfl_step_by_step_exp.py --cf selected_mnist.yaml

# 测试GACE-NO算法
python gace_no_torch_hierarchicalfl_step_by_step_exp.py --cf selected_mnist.yaml
```

## 关键特性

### 1. 信任基础联盟形成
- 使用社交网络信任关系
- 通过迭代切换形成稳定联盟
- 抵抗恶意客户端攻击

### 2. 最优集群-边缘匹配
- Kuhn-Munkres算法实现最小成本匹配
- 最小化传输成本
- 提高整体系统效率

### 3. 三层Stackelberg博弈
- 云服务器设定服务定价
- 边缘服务器确定奖励
- 客户端选择数据贡献水平
- 证明存在唯一均衡解

## 参数配置

### 系统参数
- `M`: 边缘服务器数量 (默认: 5)
- `N`: 客户端数量 (默认: 20)
- `alpha`: 信任权重参数 (默认: 0.5)

### 成本参数
- `Cn_range`: 客户端训练成本范围 (默认: 0.01-0.1)
- `Km_range`: 边缘服务器协调成本范围 (默认: 0.0-0.001)

### 博弈参数
- `a`: 边缘服务器效用系统参数 (默认: 2.3)
- `lambda_param`: 云服务器权重参数 (默认: 4.0)
- `delta_range`: 风险厌恶参数范围 (默认: 1.0-3.0)
- `theta_range`: 奖励缩放系数范围 (默认: 1.0-2.0)

## 预期结果

### 合成数据结果
- **社会效用**: GACE > GACE-NO > RA
- **云服务器效用**: GACE > GACE-NO > RA
- 随着低质量客户端增加，性能差距增大

### 真实数据结果
- **预测准确度**: GACE持续优于基线算法
- **训练损失**: GACE实现更快收敛
- **鲁棒性**: GACE在低质量客户端比例增加时保持性能

## 技术实现亮点

1. **模块化设计**: 每个算法组件独立实现，便于维护和扩展
2. **完整实验框架**: 包含所有必要的实验文件和配置
3. **自动化测试**: 提供测试脚本验证实现正确性
4. **灵活配置**: 支持多种数据集和参数配置
5. **结果可视化**: 自动生成图表和结果分析

## 文件结构
```
GACE_experiment/
├── __init__.py
├── gace_algorithm.py                    # 核心GACE算法
├── gace_torch_hierarchicalfl_step_by_step_exp.py
├── gace_ra_torch_hierarchicalfl_step_by_step_exp.py
├── gace_no_torch_hierarchicalfl_step_by_step_exp.py
├── exp1.py                             # 实验1: 社会效用分析
├── exp2.py                             # 实验2: 云服务器效用分析
├── exp3mnist.py                        # 实验3: MNIST数据集
├── exp3cifar10.py                      # 实验3: CIFAR-10数据集
├── exp3femnist.py                      # 实验3: FEMNIST数据集
├── exp3svhn.py                         # 实验3: SVHN数据集
├── exp4mnist.py                        # 实验4: MNIST低质量分析
├── exp4cifar10.py                      # 实验4: CIFAR-10低质量分析
├── exp4femnist.py                      # 实验4: FEMNIST低质量分析
├── exp4svhn.py                         # 实验4: SVHN低质量分析
├── selected_mnist.yaml                 # MNIST配置
├── selected_cifar10.yaml               # CIFAR-10配置
├── selected_femnist.yaml               # FEMNIST配置
├── selected_svhn.yaml                  # SVHN配置
├── run_gace_experiments.py             # 实验运行脚本
├── test_gace_implementation.py         # 测试脚本
├── README.md                           # 使用说明
└── IMPLEMENTATION_SUMMARY.md           # 实现总结
```

## 总结

本项目成功实现了完整的GACE FedML实验系统，包含：

- ✅ 完整的GACE算法实现
- ✅ 多个基线算法对比
- ✅ 全面的实验设计
- ✅ 自动化测试和运行脚本
- ✅ 详细的文档和说明

该实现为分层联邦学习激励机制研究提供了一个完整、可扩展的实验平台，支持多种数据集和实验场景，能够有效验证GACE算法的优越性能。

