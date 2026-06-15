# 模块文档字符串，说明本脚本用于验证所有模块是否能正确编译和导入（需要依赖安装后）
"""Verify all modules compile and import (when deps installed)."""

# 导入 os 模块，用于文件路径操作
import os
# 导入 sys 模块，用于修改 Python 模块搜索路径
import sys
# 导入 py_compile 模块，用于对 .py 文件进行语法编译检查
import py_compile

# 将项目根目录（当前文件的上上级目录）添加到系统路径中，以便导入其他模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接测试 shared/utils 模块（不需要 langchain 等外部依赖）
# 导入 safe_parse_json 函数
from shared.utils import safe_parse_json
# 断言该函数是可调用的（确保导入成功且函数存在）
assert callable(safe_parse_json)
# 输出成功的提示信息
print('[OK] shared/utils') 

# 以下模块需要 langchain 等依赖，仅在安装 requirements.txt 后才能完整导入
# 这里只验证它们的编译（语法级别），不真正导入
# 获取项目根目录路径，通过两次 dirname 回到最上层的 AI 目录
root = os.path.dirname(os.path.dirname(__file__))
# 遍历需要检查的文件列表
for f in ['shared/llm.py', 'Agent/multi_agent.py', 'Agent/main.py',
          'RAG/rag.py', 'RAG/main.py',
          'Research_assistant/research_assistant.py', 'Research_assistant/main.py']:
    # 调用 py_compile.compile 对文件进行编译检查，doraise=True 表示遇到错误会抛出异常
    py_compile.compile(os.path.join(root, f), doraise=True)
    # 编译通过则输出成功信息
    print(f'[OK] compile: {f}')

# 打印总结信息
print("\nAll checks passed.")
# 提示完整的导入测试需要先安装 requirements.txt 中的依赖
print("Note: full import test requires: pip install -r requirements.txt")