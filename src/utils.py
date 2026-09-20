"""
配置加载模块
统一读取 config/config.yaml，供项目各模块调用
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


# 项目根目录（src 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_config() -> Dict[str, Any]:
    """获取全局配置（单例模式，避免重复读取）"""
    global _CONFIG
    if "_CONFIG" not in globals():
        _CONFIG = load_config()
    return _CONFIG


if __name__ == "__main__":
    cfg = load_config()
    print("配置加载成功：")
    print(yaml.dump(cfg, allow_unicode=True, default_flow_style=False))
