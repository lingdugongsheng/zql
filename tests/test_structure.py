"""Verify all modules compile and import (when deps installed)."""
import os, sys, py_compile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test shared/utils directly (no deps needed)
from shared.utils import safe_parse_json
assert callable(safe_parse_json)
print('[OK] shared/utils')

# Modules requiring langchain will only work after: pip install -r requirements.txt
# Verify compilation (syntax-level)
root = os.path.dirname(os.path.dirname(__file__))
for f in ['shared/llm.py', 'Agent/multi_agent.py', 'Agent/main.py',
          'RAG/rag.py', 'RAG/main.py',
          'Research_assistant/research_assistant.py', 'Research_assistant/main.py']:
    py_compile.compile(os.path.join(root, f), doraise=True)
    print(f'[OK] compile: {f}')

print("\nAll checks passed.")
print("Note: full import test requires: pip install -r requirements.txt")