# MoE-RL

Mixture of Experts for Reinforcement Learning — 面向 RL rollout 长尾问题的研究项目，重点关注 code、math、puzzle 数据集上的专家混合与长尾部优化。

## 项目目标

- 探索 MoE（Mixture of Experts）架构在 RL 训练中的应用
- 针对 RL rollout 过程中的长尾问题（hard examples、rare patterns）进行专项优化
- 在 code、math、puzzle 三类基准数据集上验证方法有效性

## 目录结构

```
moe-rl/
├── src/moe_rl/          # 核心代码
│   ├── models/          # MoE 模型定义
│   ├── trainers/        # RL 训练器
│   ├── rollout/         # Rollout 与采样
│   ├── data/            # 数据集处理
│   └── utils/           # 工具函数
├── tests/               # 单元测试
├── configs/             # 配置文件
├── scripts/             # 运行脚本
└── docs/                # 文档
```

## 安装

```bash
pip install -e .
```

## 快速开始

```bash
# 训练
python scripts/train.py --config configs/default.yaml

# 评估
python scripts/evaluate.py --checkpoint path/to/checkpoint
```

## 数据集

- **Code**: HumanEval, MBPP, CodeContests
- **Math**: GSM8K, MATH, AQuA
- **Puzzle**: ARC, Big-Bench Hard

## 许可证

MIT
