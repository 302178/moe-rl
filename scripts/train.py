#!/usr/bin/env python
"""训练入口脚本。

用法:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/math_gsm8k.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moe_rl.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-RL 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件）",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式（小数据、多日志）",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件。"""
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        raise ImportError("请安装 pyyaml: pip install pyyaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")


def main():
    args = parse_args()

    logger = setup_logger("moe_rl.train", level="DEBUG" if args.debug else "INFO")

    logger.info("=" * 60)
    logger.info("MoE-RL 训练启动")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")

    config = load_config(args.config)
    logger.info(f"训练方法: {config[\'training\'][\'method\']}")
    logger.info(f"任务类型: {config[\'data\'][\'task_type\']}")
    logger.info(f"数据集: {config[\'data\'][\'dataset_name\']}")

    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    if args.debug:
        logger.info("调试模式：减少 batch size 和训练步数")
        config["training"]["per_device_train_batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 1

    logger.info("训练框架已初始化，完整训练流程待实现")
    logger.info("请参考 src/moe_rl/trainers/ 中的训练器实现")

    return 0


if __name__ == "__main__":
    sys.exit(main())
