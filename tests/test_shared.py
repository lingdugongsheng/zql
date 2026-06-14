# 测试文件的文档字符串，说明本测试只涉及 shared 模块的工具函数，不需要 LLM 依赖
"""Tests for shared utilities (no LLM dependency)."""

# 导入 os 模块，用于路径操作
import os
# 导入 sys 模块，用于修改 Python 模块搜索路径
import sys

# 将项目根目录（当前文件的上上级目录）添加到 sys.path，以便导入 shared 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 shared.utils 模块中导入 safe_parse_json 函数，并取一个简短的别名 j
from shared.utils import safe_parse_json as j

# 测试1：标准的 JSON 对象应该被正确解析
assert j('{"key":"value"}') == {"key": "value"}

# 测试2：被 Markdown 代码块包裹的 JSON 应该被提取出来并解析
assert j('```json\n{"a":1}\n```') == {"a": 1}

# 测试3：JSON 混在普通文本中时，应该能提取出花括号部分并解析
assert j('Result: {"x":[1,2,3]}.') == {"x": [1, 2, 3]}

# 测试4：当输入不是合法的 JSON 时，应该返回指定的默认值
assert j('not json', {"fallback": True}) == {"fallback": True}

# 测试5：嵌套的 JSON 对象应该被正确解析
assert j('{"outer":{"inner":"val"}}') == {"outer": {"inner": "val"}}

# 如果所有断言都通过，打印成功信息
print("test_shared: all passed")