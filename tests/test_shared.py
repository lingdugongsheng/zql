"""Tests for shared utilities (no LLM dependency)."""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('utils',
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shared', 'utils.py'))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
j = utils.safe_parse_json

assert j('{"key":"value"}') == {"key": "value"}
assert j('```json\n{"a":1}\n```') == {"a": 1}
assert j('Result: {"x":[1,2,3]}.') == {"x": [1, 2, 3]}
assert j('not json', {"fallback": True}) == {"fallback": True}
assert j('{"outer":{"inner":"val"}}') == {"outer": {"inner": "val"}}
print("test_shared: all passed")
