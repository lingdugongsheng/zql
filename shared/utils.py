# 模块文档字符串，描述本模块的主要功能
"""JSON parsing, logging, env loading."""

# 导入 os 模块，用于文件路径操作
import os
# 导入 re 模块，用于正则表达式匹配
import re
# 导入 json 模块，用于解析 JSON 数据
import json
# 导入 logging 模块，用于日志记录
import logging
# 从 typing 模块导入泛型类型，用于类型注解
from typing import Any, Dict, Optional
# 导入 dotenv 库，用于加载 .env 文件中的环境变量
import dotenv


def safe_parse_json(text: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    安全地解析 JSON 字符串。
    支持从文本中提取被 ```json ... ``` 包裹的 JSON 代码块，或直接以花括号开头的 JSON。
    如果解析失败，返回指定的默认值。
    """
    # 如果未提供默认值，使用空字典作为默认
    if default is None:
        default = {}
    # 去除输入文本的首尾空白字符
    content = text.strip()
    # 尝试匹配 Markdown 代码块：```json 或 ``` 开头，结尾为 ```
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if m:
        # 如果找到代码块，提取里面的内容（group(1) 是第一个捕获组）
        content = m.group(1).strip()
    else:
        # 如果没有代码块，尝试直接匹配以 { 开头、} 结尾的 JSON 对象（跨行匹配）
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            # 如果找到花括号包裹的内容，就使用它作为待解析的 JSON
            content = m.group(0)
    try:
        # 尝试将提取的内容解析为 JSON 并返回
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 解析失败时，记录一条警告日志，并返回默认值
        logging.getLogger(__name__).warning(f"JSON parse failed: {e}")
        return default


def setup_logging(name: str = None, level: int = logging.INFO):
    """
    配置全局日志记录器。
    name: 日志记录器的名称，如果为 None 则返回根记录器。
    level: 日志级别，默认 INFO。
    """
    # 配置基础日志格式：时间 - 名称 - 级别 - 消息
    # force=False 表示如果已有配置则不覆盖
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        force=False)
    # 返回指定名称的日志记录器，如果 name 为 None 则使用当前模块名
    return logging.getLogger(name or __name__)


def load_environment(env_path: Optional[str] = None):
    """
    加载 .env 文件中的环境变量。
    env_path: 可选，指定 .env 文件的路径；如果不提供，则自动在项目根目录查找。
    """
    if env_path:
        # 如果提供了文件路径，从指定路径加载环境变量
        dotenv.load_dotenv(env_path)
    else:
        # 否则自动搜索并加载 .env 文件
        dotenv.load_dotenv()