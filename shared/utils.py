"""JSON parsing, logging, env loading."""
import os, re, json, logging
from typing import Any, Dict, Optional
import dotenv

def safe_parse_json(text: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    if default is None:
        default = {}
    content = text.strip()
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if m:
        content = m.group(1).strip()
    else:
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            content = m.group(0)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).warning(f"JSON parse failed: {e}")
        return default

def setup_logging(name: str = None, level: int = logging.INFO):
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        force=False)
    return logging.getLogger(name or __name__)

def load_environment(env_path: Optional[str] = None):
    if env_path:
        dotenv.load_dotenv(env_path)
    else:
        dotenv.load_dotenv()
