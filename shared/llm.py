# 模块文档字符串，简要说明本模块的功能
"""Thread-safe LLM cache with retry support."""

# 导入 os 模块，用于读取环境变量（获取 API 密钥和基础 URL）
import os
# 导入 time 模块，用于在重试之间添加延迟
import time
# 导入 threading 模块，用于创建线程锁，保证模型实例的线程安全创建
import threading


# 定义一个模型缓存类，用于延迟创建并缓存 LLM 实例，同时确保线程安全
class ModelCache:
    # 构造函数，初始化缓存参数
    # model: 要使用的模型名称，默认为 deepseek-chat
    # temperature: 模型温度，控制生成随机性，默认 0.3
    # max_tokens: 模型最大输出 token 数，默认 1000
    def __init__(self, model="deepseek-chat", temperature=0.3, max_tokens=1000):
        # 内部变量 _model 用于存储已创建的模型实例，初始为 None 表示尚未创建
        self._model = None
        # 创建一个线程锁，用于保护模型创建时的并发访问
        self._lock = threading.Lock()
        # 保存传入的模型名称
        self._model_name = model
        # 保存温度参数
        self._temperature = temperature
        # 保存最大 token 数
        self._max_tokens = max_tokens

    # 获取模型实例的方法，如果已存在则直接返回，否则创建新的实例（双重检查锁定）
    def get(self):
        # 第一次检查：如果模型已经创建，直接返回，避免不必要的锁竞争
        if self._model is not None:
            return self._model
        # 获取锁，确保同一时间只有一个线程进入创建代码块
        with self._lock:
            # 第二次检查：在获取锁后再次检查，因为可能在等待锁时其他线程已经创建了模型
            if self._model is not None:
                return self._model
            # 从环境变量中获取 DeepSeek API 密钥
            api_key = os.getenv("DEEPSEEK_API_KEY")
            # 从环境变量中获取 DeepSeek API 基础 URL
            base_url = os.getenv("DEEPSEEK_BASE_URL")
            # 如果密钥或 URL 未设置，抛出运行时错误，提示用户配置
            if not api_key or not base_url:
                raise RuntimeError("DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL must be set")
            # 延迟导入 ChatOpenAI，避免在不需要时加载依赖
            from langchain_openai import ChatOpenAI  # lazy import
            # 创建 ChatOpenAI 实例，传入模型名称、温度、最大 token 数、API 密钥和基础 URL
            self._model = ChatOpenAI(
                model=self._model_name, temperature=self._temperature,
                max_tokens=self._max_tokens, api_key=api_key, base_url=base_url)
            # 返回新创建的模型实例
            return self._model


# 定义一个带重试机制的 LLM 调用函数
# model_or_cache: 可以是 ModelCache 实例，也可以是直接的 ChatOpenAI 实例
# messages: 要发送给 LLM 的消息列表
# max_retries: 最大重试次数，默认 3
# delay: 每次重试之间的等待时间（秒），默认 1.5
def llm_invoke_with_retry(model_or_cache, messages, max_retries=3, delay=1.5):
    # 如果传入的是 ModelCache 实例，则调用 .get() 获取实际的 LLM 对象；否则直接使用传入的 LLM
    llm = model_or_cache.get() if hasattr(model_or_cache, 'get') else model_or_cache
    # 循环尝试调用，从第 1 次到第 max_retries 次
    for attempt in range(1, max_retries + 1):
        try:
            # 调用 LLM 的 invoke 方法，传入消息并等待响应
            return llm.invoke(messages)
        except Exception as e:
            # 如果发生异常，动态获取当前模块的日志记录器
            logger = __import__('logging').getLogger(__name__)
            # 记录警告日志，显示当前尝试次数和错误信息
            logger.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
            # 如果已经是最后一次尝试，重新抛出异常，不再重试
            if attempt == max_retries:
                raise
            # 否则等待指定的延迟时间，然后继续下一次循环
            time.sleep(delay)
