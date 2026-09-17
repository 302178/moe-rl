# 架构设计

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        训练入口 (scripts/)                    │
│                   train.py / evaluate.py                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      训练器 (trainers/)                        │
│         ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│         │ PPO      │    │ GRPO     │    │ DPO      │       │
│         └────┬─────┘    └────┬─────┘    └──────────┘       │
└──────────────┼────────────────┼──────────────────────────────┘
               │                │
    ┌──────────▼────┐   ┌──────▼──────────┐
    │ Rollout 生成   │   │ 奖励计算         │
    │ (rollout/)     │   │ (utils/reward)  │
    └──────────┬────┘   └──────────────────┘
               │
    ┌──────────▼────────────────────────────┐
    │          模型 (models/)                 │
    │   ┌─────────┐  ┌────────┐  ┌───────┐ │
    │   │ MoELayer│  │ Router │  │ LLM   │ │
    │   └─────────┘  └────────┘  └───────┘ │
    └─────────────────────────────────────────┘
               │
    ┌──────────▼────────────────────────────┐
    │        数据 (data/)                     │
    │  Code / Math / Puzzle 数据集处理        │
    └─────────────────────────────────────────┘
```

## 核心模块

### 1. models/ - 模型定义

- **MoELayer**: Mixture of Experts 层，支持 top-k 路由、负载均衡损失
- **Router**: 多种路由策略（TopK、Expert Choice）
- 支持将 MoE 层插入到现有 LLM 中

### 2. trainers/ - 训练器

- **PPOTrainer**: 经典 PPO 算法，支持 value network
- **GRPOTrainer**: Group Relative Policy Optimization，无需 critic，适合数学推理
- 统一接口，支持 MoE 负载均衡损失

### 3. rollout/ - Rollout 与采样

- **RolloutGenerator**: 批量生成管理
- **LongTailSampler**: 长尾样本采样器，维护 hard example 池，按难度加权采样

### 4. data/ - 数据集

- **CodeDataset**: HumanEval、MBPP、CodeContests、APPS
- **MathDataset**: GSM8K、MATH、AQuA、SVAMP
- **PuzzleDataset**: ARC、Big-Bench Hard
- 统一的答案提取和验证接口

### 5. utils/ - 工具

- **metrics**: Pass@k、准确率等指标计算
- **reward**: 代码执行奖励、数学答案奖励、格式奖励
- **logging**: 统一日志配置

## 长尾问题解决方案

### 问题
RL rollout 中存在大量 easy examples，模型在这些样本上已经很好，
但 hard examples（长尾）得不到充分训练。

### 方案
1. **难度估计**: 基于 reward/loss 估计样本难度
2. **Hard Pool**: 维护 hard example 池，按难度加权采样
3. **MoE 专家分工**: 不同专家专注不同难度/类型的样本
4. **课程学习**: 从易到难，逐步增加 hard example 比例

## 数据流

```
数据集 → 难度估计 → [Hard Pool] [Regular Pool]
                          ↓
                    混合采样 (hard_ratio)
                          ↓
                    Rollout 生成
                          ↓
                    奖励计算
                          ↓
                    PPO/GRPO 更新
                          ↓
                    更新 Hard Pool
```
