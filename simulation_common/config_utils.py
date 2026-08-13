"""各入口共用的YAML读取函数。"""

import yaml


def load_yaml(config_file):
    """以UTF-8读取一个YAML文件。"""
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
