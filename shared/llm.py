"""Thread-safe LLM cache with retry support."""
import os, time, threading


class ModelCache:
    def __init__(self, model="deepseek-chat", temperature=0.3, max_tokens=1000):
        self._model = None
        self._lock = threading.Lock()
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL")
            if not api_key or not base_url:
                raise RuntimeError("DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL must be set")
            from langchain_openai import ChatOpenAI  # lazy import
            self._model = ChatOpenAI(
                model=self._model_name, temperature=self._temperature,
                max_tokens=self._max_tokens, api_key=api_key, base_url=base_url)
            return self._model


def llm_invoke_with_retry(model_or_cache, messages, max_retries=3, delay=1.5):
    """Invoke LLM with retry on failure. Accepts ModelCache or ChatOpenAI."""
    llm = model_or_cache.get() if hasattr(model_or_cache, 'get') else model_or_cache
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            time.sleep(delay)
