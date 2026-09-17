#!/usr/bin/env python
"""评估入口脚本。

用法:
    python scripts/evaluate.py --checkpoint path/to/model --config configs/default.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe_rl.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-RL 评估")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="评估数据集 split",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="评估样本数量",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="评估结果输出文件 (JSON)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件。"""
    try:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("请安装 pyyaml: pip install pyyaml")


def main():
    args = parse_args()

    logger = setup_logger("moe_rl.eval", level="INFO")

    logger.info("=" * 60)
    logger.info("MoE-RL 评估启动")
    logger.info("=" * 60)
    logger.info(f"模型检查点: {args.checkpoint}")
    logger.info(f"配置文件: {args.config}")

    config = load_config(args.config)
    task_type = config["data"]["task_type"]
    logger.info(f"任务类型: {task_type}")

    if args.eval_split:
        config["data"]["eval_split"] = args.eval_split
    if args.num_samples:
        config["eval"]["num_eval_samples"] = args.num_samples

    logger.info("评估框架已初始化，完整评估流程待实现")
    logger.info(f"任务类型: {task_type}")
    logger.info(f"评估指标: {config[\'eval\'][\'metrics\']}")

    results = {
        "checkpoint": args.checkpoint,
        "task_type": task_type,
        "metrics": {m: 0.0 for m in config["eval"]["metrics"]},
        "num_samples": config["eval"]["num_eval_samples"],
    }

    if args.output_file:
        import json

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
