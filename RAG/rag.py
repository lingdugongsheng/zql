# %%
# ===================第一部分：导入必要的库===================
import os                          # 操作系统接口，用于读取环境变量、文件路径操作
import re                          # 正则表达式，用于从文本中提取数字（如置信度评分）
import uuid                        # 生成全局唯一标识符，用于增量索引时的文档ID
import hashlib                     # 哈希算法，计算文本MD5，用于精确去重
import shutil                      # 高级文件操作，用于删除整个向量存储目录
import logging                     # 日志模块，替代print输出，记录程序运行信息
import sys
from dataclasses import dataclass  # 装饰器，简化配置类的定义，自动生成__init__等方法
from typing import List, Dict, Any, Optional, TypedDict, Set  # 类型注解，提高代码可读性和可维护性

import chromadb                    # Chroma 底层客户端，用于列出持久化集合
import numpy as np                 # 数值计算库，用于向量相似度计算和矩阵操作
from langchain_core.messages import HumanMessage, AIMessage  # 人类和AI消息类型，用于构建对话历史
from langchain_core.output_parsers import StrOutputParser    # 字符串输出解析器，提取模型返回的纯文本
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示模板和对话历史占位符
from langchain_chroma import Chroma                           # Chroma 向量数据库客户端
from langchain_core.documents import Document                 # 文档对象，表示文本块及其元数据
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归文本分割器，用于文档切片
from langgraph.constants import START, END                    # LangGraph 图中的特殊节点：开始和结束
from langgraph.graph import StateGraph                        # LangGraph 状态图，用于构建有状态的工作流

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import setup_logging, load_environment, ModelCache, llm_invoke_with_retry

# %%
# ===================第二部分：加载和验证环境变量===================
load_environment()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")     # 读取DeepSeek的API密钥
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")   # 读取DeepSeek的API基础URL
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")     # 读取智谱AI的API密钥
ZHIPUAI_BASE_URL = os.getenv("ZHIPUAI_BASE_URL")

# 验证必要的环境变量是否存在，缺失则抛出错误并终止程序
if not DEEPSEEK_API_KEY:
    raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置，请在 .env 文件中添加该变量。")
if not DEEPSEEK_BASE_URL:
    raise ValueError("环境变量 DEEPSEEK_BASE_URL 未设置，请在 .env 文件中添加该变量。")

# 配置日志
logger = setup_logging(__name__)

# 模型和嵌入模型的全局缓存，避免重复初始化，节省资源和时间
_model_cache = ModelCache(temperature=0.3, max_tokens=1000)
_EMBEDDINGS_CACHE = None

# %%
# ===================第三部分：工具函数===================
def get_model():
    """Get the shared model instance."""
    return _model_cache.get()

def get_embeddings():
    """
    获取文本嵌入模型的单例实例。
    优先使用智谱AI的 embedding-2 模型，若不可用则回退到本地 HuggingFace 模型。
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE
    try:
        if ZHIPUAI_API_KEY:   # 如果有智谱API密钥，使用其嵌入模型
            from langchain_openai import OpenAIEmbeddings
            _EMBEDDINGS_CACHE = OpenAIEmbeddings(
                model="embedding-2",          # 智谱提供的嵌入模型名称
                api_key=ZHIPUAI_API_KEY,
                base_url=ZHIPUAI_BASE_URL
            )
        else:                  # 否则使用本地 HuggingFace 模型作为后备
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # 轻量级本地模型
            )
        return _EMBEDDINGS_CACHE
    except Exception as e:
        raise RuntimeError(f"无法加载嵌入模型: {e}")

def list_persisted_collections(persist_directory: str) -> List[str]:
    """
    列出指定持久化目录下所有的 Chroma 集合名称。
    使用 chromadb 客户端获取准确的集合列表。
    """
    if not persist_directory or not os.path.exists(persist_directory):
        return []
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        return [col.name for col in client.list_collections()]
    except Exception as e:
        logger.error(f"列出集合失败：{e}")
        return []

def get_collection_stats(vector_store: Chroma) -> Dict[str, Any]:
    """
    获取 Chroma 向量存储的基本统计信息：文档数量、ID列表、文档内容。
    注意：对于大型数据库，获取全部数据可能消耗内存，仅用于演示或小型索引。
    """
    try:
        collection_data = vector_store.get()   # 获取集合中的所有数据（可能很重）
        return {
            "count": len(collection_data["ids"]),
            "ids": collection_data["ids"],
            "documents": collection_data["documents"]
        }
    except Exception as e:
        logger.error(f"获取集合数据失败：{e}")
        return {"count": 0, "ids": [], "documents": []}

# %%
# ===================第四部分：RAG配置类===================
@dataclass
class RAGConfig:
    """RAG 系统全局配置，使用 dataclass 管理参数和默认值。"""
    temperature: float = 0.3     # 模型生成温度，控制随机性
    max_tokens: int = 1000       # 模型单次最大输出token数
    chunk_size: int = 500        # 文档分块大小（字符数）
    chunk_overlap: int = 100     # 相邻文本块重叠的字符数，保持上下文连贯
    dedup_threshold: float = 0.9 # 语义去重时的相似度阈值（0-1之间）
    deduplicate: bool = True     # 是否启用文档去重
    top_k: int = 3               # 检索时返回的最相关文档数量
    search_type: str = "similarity"  # 检索类型（目前使用相似度搜索）

# %%
# ===================第五部分：RAG状态定义===================
class RAGState(TypedDict):
    """
    定义 RAG 工作流中各个节点共享的状态。
    使用 TypedDict 提供类型提示，便于在 LangGraph 中传递。
    """
    query: str                              # 当前用户问题（可能被改写）
    chat_history: List[Dict[str, str]]      # 对话历史，格式 [{"role": "user", "content": "..."}]
    documents: List[Document]               # 检索到的相关文档列表
    context: str                            # 拼接后的上下文文本，供生成器使用
    answer: str                             # 最终生成的回答
    sources: List[Dict[str, str]]           # 回答引用的来源信息
    confidence: float                       # 生成回答的置信度评分

# %%
# ===================第六部分：文档去重器===================
class DocumentDeduplicator:
    """
    文档去重器：支持精确去重（MD5哈希）和语义去重（向量相似度）。
    避免重复文档影响检索质量。
    """
    def __init__(self, embeddings=None, similarity_threshold: float = 0.95):
        self.embeddings = embeddings                  # 嵌入模型，用于计算语义向量
        self.similarity_threshold = similarity_threshold  # 语义去重阈值
        self.seen_hashes: Set[str] = set()            # 已见过的文档哈希值，用于精确去重

    @staticmethod
    def _normalize_text(text: str) -> str:
        """规范化文本：去除多余空白，让哈希更稳定。"""
        return " ".join(text.strip().split())

    @staticmethod
    def _compute_hash(text: str) -> str:
        """计算文本的 MD5 哈希，用于精确匹配去重。"""
        normalized = DocumentDeduplicator._normalize_text(text)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _batch_compute_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        批量计算文本的嵌入向量。
        如果嵌入模型不可用或失败，返回 None。
        """
        if not self.embeddings:
            return None
        try:
            # 优先使用 embed_documents 方法（更高效）
            if hasattr(self.embeddings, "embed_documents"):
                return self.embeddings.embed_documents(texts)
            # 否则逐条调用 embed_query
            return [self.embeddings.embed_query(text) for text in texts]
        except (ValueError, RuntimeError) as e:
            logger.error(f"嵌入计算失败：{e}")
            return None

    def _semantic_deduplicate(self, documents: List[Document], embeddings: List[List[float]]) -> List[Document]:
        """
        基于嵌入向量相似度进行语义去重。
        使用贪心策略：对于每个文档，仅当它与已保留的所有文档都不相似时才保留。
        这可以避免相似链导致的过度去重。
        """
        vectors = np.array(embeddings)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        valid_mask = norms.flatten() > 1e-10        # 过滤零向量（无效向量）
        if not np.any(valid_mask):
            return documents                        # 如果没有有效向量，返回原文档

        valid_indices = np.where(valid_mask)[0]     # 有效文档在列表中的索引
        valid_vectors = vectors[valid_indices] / norms[valid_indices]  # 归一化向量

        keep_indices = []   # 存储保留的文档在 valid_vectors 中的位置
        for idx, orig_idx in enumerate(valid_indices):
            vec = valid_vectors[idx]
            should_keep = True
            # 检查当前文档与所有已保留文档的相似度
            for kept_idx in keep_indices:
                kept_vec = valid_vectors[kept_idx]
                similarity = np.dot(vec, kept_vec)   # 余弦相似度（向量已归一化）
                if similarity >= self.similarity_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep_indices.append(idx)

        # 映射回原始文档索引并返回文档
        final_indices = [valid_indices[idx] for idx in keep_indices]
        return [documents[i] for i in final_indices]

    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        主去重方法：先精确去重（哈希），再语义去重。
        返回去重后的文档列表。
        """
        if not documents:
            return []
        self.seen_hashes.clear()      # 清空哈希集合

        # 第一步：基于 MD5 哈希的精确去重
        hash_unique = []
        for doc in documents:
            text = doc.page_content
            if not text or not text.strip():   # 跳过空文档
                continue
            doc_hash = self._compute_hash(text)
            if doc_hash in self.seen_hashes:   # 已存在相同哈希，跳过
                continue
            self.seen_hashes.add(doc_hash)
            hash_unique.append(doc)

        # 如果只剩一个文档或没有嵌入模型，则无需语义去重
        if len(hash_unique) <= 1 or not self.embeddings:
            return hash_unique

        # 第二步：语义去重
        texts = [doc.page_content for doc in hash_unique]
        embeddings = self._batch_compute_embeddings(texts)
        if embeddings is None or len(embeddings) != len(hash_unique):
            return hash_unique   # 嵌入计算失败，返回精确去重后的结果

        return self._semantic_deduplicate(hash_unique, embeddings)

# %%
# ===================第七部分：文档处理器===================
class DocumentProcessor:
    """
    文档处理器：负责文档加载、分割、去重以及向量存储的创建与加载。
    """
    def __init__(self, config: RAGConfig, persist_directory: Optional[str] = None):
        self.config = config
        self.persist_directory = persist_directory   # 向量存储持久化目录
        # 初始化文本分割器，使用常见的中英文标点作为分割符
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.embeddings = get_embeddings()          # 获取嵌入模型实例
        self.vector_store = None                    # 向量存储实例，稍后创建或加载
        self.deduplicator = DocumentDeduplicator(
            embeddings=self.embeddings,
            similarity_threshold=config.dedup_threshold
        )

    @staticmethod
    def load_documents(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """
        将原始文本列表转换为 LangChain Document 对象列表。
        可附带元数据，如来源、主题等。
        """
        documents = []
        for i, text in enumerate(texts):
            # 如果没有提供对应的元数据，则创建一个默认来源
            metadata = metadatas[i] if metadatas is not None and i < len(metadatas) else {'source': f"doc_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """使用文本分割器将长文档拆分成较小的块，便于向量化和检索。"""
        return self.text_splitter.split_documents(documents)

    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """对文档块执行去重（根据配置决定是否执行）。"""
        if not self.config.deduplicate:
            logger.info("跳过文档去重")
            return documents
        logger.info("执行文档去重")
        return self.deduplicator.deduplicate_documents(documents)

    def create_vector_store(self, documents: List[Document], collection_name: str = "default") -> Chroma:
        """
        从文档列表创建 Chroma 向量存储。
        根据是否配置了持久化目录，决定存储方式（内存或磁盘）。
        """
        storage_type = "持久化" if self.persist_directory else "内存"
        logger.info(f"创建{storage_type}向量存储")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,   # 若为None则存储在内存
            collection_name=collection_name
        )
        logger.info("向量存储创建成功")
        return self.vector_store

    def load_existing_vector_store(self, collection_name: str = "enhanced_docs") -> Optional[Chroma]:
        """
        从磁盘加载已存在的向量存储。
        仅当设置了持久化目录且目录存在时有效。
        """
        if not self.persist_directory:
            logger.info("未配置持久化目录，无法加载向量存储")
            return None
        if not os.path.exists(self.persist_directory):
            logger.warning(f"向量存储目录不存在: {self.persist_directory}")
            return None
        try:
            logger.info(f"正在加载向量存储 '{collection_name}'")
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            stats = get_collection_stats(self.vector_store)
            logger.info(f"向量存储 '{collection_name}' 加载成功")
            if stats["count"] == 0:
                logger.warning(f"向量存储 '{collection_name}' 中没有文档")
                return None
            return self.vector_store
        except Exception as e:
            logger.error(f"加载向量存储失败：{e}")
            return None

    def list_collections(self) -> List[str]:
        """列出持久化目录下可用的所有集合名称。"""
        return list_persisted_collections(self.persist_directory)

    def process(self, texts: List[str], metadatas: Optional[List[Dict]] = None,
                collection_name: str = "default") -> Chroma:
        """
        完整的文档处理流程：加载、分割、去重、创建向量存储。
        返回创建好的向量存储对象。
        """
        logger.info("=" * 50)
        logger.info("开始文档处理流程")
        logger.info("=" * 50)

        logger.info("步骤1：加载文档")
        documents = self.load_documents(texts, metadatas)
        logger.info(f"加载完成，文档数量: {len(documents)}")

        logger.info("步骤2：分割文档")
        chunks = self.split_documents(documents)
        logger.info(f"分割完成，文档块数量: {len(chunks)}")

        if self.config.deduplicate:
            logger.info("步骤3：去重文档")
            chunks = self.deduplicate(chunks)
            logger.info(f"去重完成，剩余文档块数量: {len(chunks)}")
        else:
            logger.info("步骤3：跳过文档去重")

        logger.info("步骤4：创建向量存储")
        vector_store = self.create_vector_store(chunks, collection_name)
        stats = get_collection_stats(vector_store)
        logger.info(f"向量储存创建完成，文档数量={stats['count']}")
        return vector_store

# %%
# ===================第八部分：检索器===================
class Retriever:
    """检索器：封装从向量存储中检索相关文档的方法。"""
    def __init__(self, vector_store: Chroma, config: RAGConfig, collection_name: str = "default"):
        self.vector_store = vector_store
        self.config = config
        self.collection_name = collection_name

    def retrieve(self, query: str, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        标准相似度检索：返回与查询最相关的 top_k 个文档。
        可选的 filter_dict 用于按元数据过滤。
        """
        docs = self.vector_store.similarity_search(
            query=query,
            k=self.config.top_k,      # 返回最相关的前 top_k 个文档
            filter=filter_dict
        )
        logger.info(f"检索完成，找到 {len(docs)} 个相关文档")
        return docs

# %%
# ===================第九部分：生成器===================
class Generator:
    """
    生成器：负责查询改写、基于上下文生成回答以及置信度评估。
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = get_model()    # 获取语言模型实例

        # 定义 RAG 回答提示模板
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的问答助手，请基于提供的上下文信息回答用户的问题。
        重要规则：
        1. 必须使用提供的上下文信息来回答问题。
        2. 如果上下文中没有相关信息，请诚实地说"根据提供的信息，我无法回答。"
        3. 回答要准确、简洁、有条理。
        4. 在回答末尾标注信息来源。

        上下文信息：
        {context}
        """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),  # 对话历史占位符（可选）
            ("human", "{query}")
        ])

        # 定义查询改写提示模板
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。请根据对话历史，将用户的问题改写为一个独立、完整的查询。
        如果问题本身已经很清晰完整，直接返回原问题。
        只返回改写后的查询，不要添加任何解释。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "原始问题：{query}\n\n请改写为独立完整的查询：")
        ])

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        根据对话历史改写查询，解决指代消解和信息省略问题。
        例如“它的特点”会根据上文改写成“LangChain的特点”。
        """
        if not chat_history:
            return query   # 如果没有历史，直接返回原问题

        # 将字典格式的历史转换为 LangChain 消息对象列表，只取最近6条
        messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
            for msg in chat_history[-6:]
        ]
        chain = self.rewrite_prompt | self.llm | StrOutputParser()
        return llm_invoke_with_retry(chain, {"query": query, "chat_history": messages})

    def generate(self, query: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        基于检索到的上下文和对话历史生成最终回答。
        """
        messages = []
        if chat_history:
            for msg in chat_history[-6:]:   # 同样只取最近6条
                messages.append(
                    HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                )
        chain = self.rag_prompt | self.llm | StrOutputParser()
        return llm_invoke_with_retry(chain, {
            "query": query,
            "chat_history": messages,
            "context": context
        })

    def evaluate(self, query: str, context: str, answer: str) -> float:
        """
        评估生成回答的质量，返回 0 到 1 之间的置信度分数。
        基于回答是否忠实于上下文。
        """
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严格的评估专家。请根据以下标准评估回答质量：
        评分标准（0-1分）：
        请仔细检查回答是否严格基于提供的上下文，然后返回一个精确的分数。
        只返回数字，不要任何解释。"""),
            ("human", """上下文：
        {context}
        问题：
        {query}
        回答：
        {answer}
        请评分（0-1）：""")
        ])
        chain = eval_prompt | self.llm | StrOutputParser()
        try:
            response = llm_invoke_with_retry(chain, {
                "context": context,
                "query": query,
                "answer": answer
            }).strip()
            # 尝试提取数字
            if response.replace('.', '', 1).isdigit():
                score = float(response)
            else:
                match = re.search(r'[\d.]+', response)
                score = float(match.group()) if match else 0.3
            return min(max(score, 0), 1.0)   # 确保分数在 0-1 之间
        except Exception as e:
            logger.error(f"置信度评估失败: {e}")
            return 0.3   # 异常时返回默认保守分数

# %%
# ===================第十部分：RAG链===================
class RAGChain:
    """
    RAG 链主控制器：整合文档处理、检索、生成，并基于 LangGraph 构建有状态工作流。
    提供索引文档、查询、管理历史等功能。
    """
    def __init__(self, config: RAGConfig = None, persist_directory: Optional[str] = None):
        self.config = config or RAGConfig()   # 使用传入配置或默认配置
        self.persist_directory = persist_directory
        self.processor = DocumentProcessor(config=self.config, persist_directory=persist_directory)
        self.retriever = None                 # 检索器将在索引或加载时初始化
        self.generator = Generator(config=self.config)
        self.graph = None                     # LangGraph 状态图，编译后用于执行工作流
        self.chat_history = []                # 保存全局对话历史
        self.collection_name = "default"      # 当前使用的集合名称
        self.document_count = 0               # 当前索引的文档块数量

    def clear_history(self):
        """清空对话历史，开始新的会话。"""
        self.chat_history = []
        logger.info("对话历史已清除")

    def index_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                        collection_name: str = "default"):
        """
        从原始文本列表创建新索引。
        会执行完整的文档处理流程，并构建检索器和工作流图。
        """
        self.collection_name = collection_name
        vector_store = self.processor.process(texts, metadatas, collection_name)
        self.retriever = Retriever(vector_store, self.config, collection_name)
        self._build_graph()    # 构建 LangGraph 工作流
        stats = get_collection_stats(vector_store)
        self.document_count = stats['count']
        info = {"name": collection_name, "count": self.document_count, "metadata": {}}
        logger.info(f"索引统计: {info}")
        if self.config.deduplicate:
            logger.info(f"已启用文档去重 (阈值: {self.config.dedup_threshold})")

    def load_existing_index(self, collection_name: str = "enhanced_docs") -> bool:
        """
        从磁盘加载已有的向量索引。
        成功返回 True，失败返回 False。
        """
        if not self.persist_directory:
            logger.warning("内存模式下无法加载已存在的索引")
            return False
        vector_store = self.processor.load_existing_vector_store(collection_name)
        if vector_store:
            self.retriever = Retriever(vector_store, self.config, collection_name)
            self.collection_name = collection_name
            self._build_graph()
            stats = get_collection_stats(vector_store)
            self.document_count = stats['count']
            info = {"name": collection_name, "count": self.document_count, "metadata": {}}
            logger.info(f"已加载索引: {info}")
            if self.config.deduplicate:
                logger.info("已启用文档去重")
            return True
        return False

    def delete_collection(self, collection_name: str = "default"):
        """删除持久化的集合（仅限磁盘模式）。"""
        if not self.persist_directory:
            logger.warning("内存模式下无法删除集合")
            return
        try:
            collection_path = os.path.join(self.persist_directory, collection_name)
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
                logger.info(f"集合 '{collection_name}' 已删除")
            if self.collection_name == collection_name:
                self.retriever = None
                self.collection_name = "default"
                self.document_count = 0
        except Exception as e:
            logger.error(f"删除集合失败: {e}")

    def list_collections(self) -> List[str]:
        """列出所有持久化集合名称。"""
        return list_persisted_collections(self.persist_directory)

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """向现有索引增量添加文档（不重新创建索引）。返回新增块数。"""
        if not self.retriever:
            raise ValueError("请先加载或创建索引")
        logger.info("增量索引：添加新文档")
        documents = self.processor.load_documents(texts, metadatas)
        chunks = self.processor.split_documents(documents)
        if self.processor.config.deduplicate:
            chunks = self.processor.deduplicate(chunks)
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        self.retriever.vector_store.add_documents(documents=chunks, ids=ids)
        stats = get_collection_stats(self.retriever.vector_store)
        self.document_count = stats['count']
        chunk_count = len(chunks)
        logger.info(f"已添加 {chunk_count} 个新文档块，当前总文档数: {self.document_count}")
        return chunk_count

    def _build_graph(self):
        """
        构建 LangGraph 状态图，定义 RAG 工作流：
        1. process_query: 改写查询（如果有历史）
        2. retrieve: 检索相关文档
        3. generate: 生成回答
        4. evaluate: 评估置信度
        """
        # 节点函数：每个函数接收并返回 RAGState
        def process_query(state: RAGState) -> RAGState:
            if state.get("chat_history"):
                rewritten = self.generator.rewrite_query(state["query"], state["chat_history"])
                logger.debug(f"查询改写：{state['query']} -> {rewritten}")
                state["query"] = rewritten
            return state

        def retrieve_documents(state: RAGState) -> RAGState:
            docs = self.retriever.retrieve(state["query"])
            logger.info(f"检索到 {len(docs)} 个相关文档")
            state["documents"] = docs
            # 拼接上下文，并给每个文档加上编号
            state["context"] = "\n\n".join(f"[文档{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))
            state["sources"] = [
                {
                    "index": i + 1,
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:100] + "..."
                }
                for i, doc in enumerate(docs)
            ]
            return state

        def generate_answer(state: RAGState) -> RAGState:
            state["answer"] = self.generator.generate(
                query=state["query"],
                context=state["context"],
                chat_history=state.get("chat_history", []),
            )
            logger.info("生成回答完成")
            return state

        def evaluate_response(state: RAGState) -> RAGState:
            state["confidence"] = self.generator.evaluate(
                query=state["query"],
                context=state["context"],
                answer=state["answer"]
            )
            logger.info(f"置信度评估：{state['confidence']:.2f}")
            return state

        # 创建状态图
        graph = StateGraph(RAGState)
        # 添加节点
        graph.add_node("process_query", process_query)
        graph.add_node("retrieve", retrieve_documents)
        graph.add_node("generate", generate_answer)
        graph.add_node("evaluate", evaluate_response)
        # 定义边：开始 -> 查询处理 -> 检索 -> 生成 -> 评估 -> 结束
        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", END)
        self.graph = graph.compile()   # 编译为可执行的应用

    def query(self, question: str) -> Dict[str, Any]:
        """
        执行一次 RAG 查询：输入问题，返回包含回答、来源和置信度的字典。
        内部会调用编译好的 LangGraph 工作流。
        """
        if not self.retriever:
            raise ValueError("请先调用 index_documents() 或 load_existing_index()")
        # 简单的输入验证
        if not question or not question.strip():
            raise ValueError("问题不能为空")
        if len(question) > 1000:
            raise ValueError("问题过长，请缩短后再试")
        question = question.strip()
        logger.info(f"处理问题：{question}")

        # 构建初始状态
        initial_state = {
            "query": question,
            "chat_history": self.chat_history,
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
        }
        result = self.graph.invoke(initial_state)   # 执行工作流
        # 更新全局对话历史
        self.chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]}
        ])
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        }

# %%
# ==================== 第十二部分：测试文档和主程序 ====================
# 示例文档数据，用于快速测试 RAG 系统
SAMPLE_DOCUMENTS = [
    {
        "text": """LangChain 简介

LangChain 是一个用于开发大型语言模型（LLM）应用的开源框架。它提供了一套标准化的接口和工具，
帮助开发者快速构建基于 LLM 的应用程序。

主要特点：
1. 模块化设计：所有组件都可以独立使用或组合使用
2. 链式调用：支持将多个组件链接在一起形成复杂的工作流
3. 记忆管理：内置多种记忆类型，支持对话历史管理
4. 工具集成：可以轻松集成外部工具和 API

LangChain 1.0 于 2025 年 10 月发布，带来了重大改进：
- 更清晰的 API 设计
- 更好的类型提示支持
- 改进的错误处理
- 与 LangGraph 的深度集成

使用场景包括：聊天机器人、问答系统、文档分析、代码生成等。""",
        "metadata": {"source": "langchain_intro.txt", "topic": "introduction"}
    },
    {
        "text": """LangGraph 介绍

LangGraph 是 LangChain 生态系统中的一个重要组件，专门用于构建有状态的、多步骤的 AI 应用。
它基于图结构来定义工作流，使得复杂的 AI 流程变得清晰和可控。

核心概念：
1. 状态（State）：使用 TypedDict 定义应用状态，在节点间传递
2. 节点（Node）：处理状态的函数，执行具体的业务逻辑
3. 边（Edge）：定义节点之间的连接和流转规则
4. 条件边：根据状态动态决定下一个节点

LangGraph 的优势：
- 可视化流程：图结构使工作流一目了然
- 状态管理：自动处理状态的传递和更新
- 检查点：支持中间状态的保存和恢复
- 人机协作：支持 human-in-the-loop 模式

典型应用场景：
- 多步骤推理
- 多代理协作
- 复杂决策流程
- 带有循环的工作流""",
        "metadata": {"source": "langgraph_intro.txt", "topic": "langgraph"}
    },
    {
        "text": """RAG（检索增强生成）原理

RAG 是一种结合检索和生成的技术，通过从知识库中检索相关信息来增强 LLM 的回答质量。

工作流程：
1. 文档处理：将文档分割成小块，并转换为向量表示
2. 向量存储：将文档向量存入向量数据库
3. 查询检索：用户提问时，检索最相关的文档块
4. 上下文增强：将检索到的内容作为上下文提供给 LLM
5. 回答生成：LLM 基于上下文生成准确的回答

RAG 的优势：
- 减少幻觉：基于真实文档生成回答
- 知识更新：无需重新训练模型即可更新知识
- 来源可追溯：可以引用具体的信息来源
- 成本效益：比微调模型更经济

最佳实践：
- 选择合适的分块策略
- 优化检索算法
- 设计有效的提示模板
- 实现结果重排序""",
        "metadata": {"source": "rag_principles.txt", "topic": "rag"}
    },
    {
        "text": """向量数据库介绍

向量数据库是专门用于存储和检索向量数据的数据库系统，是 RAG 系统的核心组件之一。

主要特点：
1. 高效相似度搜索：支持快速的近似最近邻（ANN）搜索
2. 可扩展性：能够处理数百万甚至数十亿级别的向量
3. 实时更新：支持动态添加和删除向量
4. 元数据过滤：支持基于元数据的过滤查询

常见的向量数据库：
- Chroma：轻量级，适合开发和原型
- Pinecone：云原生，完全托管
- Milvus：开源，高性能
- Weaviate：支持混合搜索
- FAISS：Facebook 开发，适合研究

选择建议：
- 开发阶段：使用 Chroma 或内存向量存储
- 生产环境：根据规模选择 Pinecone 或 Milvus
- 需要混合搜索：考虑 Weaviate

性能优化：
- 选择合适的索引类型
- 调整搜索参数
- 使用批量操作""",
        "metadata": {"source": "vector_db.txt", "topic": "database"}
    }
]

# %%
# ==================== 第十三部分：主程序 ====================
def main():
    """主程序：演示 RAG 系统的使用流程，包括索引、单轮问答、多轮对话和清除历史。"""
    print("=" * 60)                     # 打印分隔线，美化输出
    print("RAG 检索增强生成系统演示")   # 显示演示标题
    print("=" * 60)                     # 打印分隔线

    # 初始化 RAG 系统（自定义配置）
    print("\n初始化 RAG 系统...")        # 提示用户正在初始化系统
    config = RAGConfig(
        chunk_size=300,                # 设置分块大小为300字符
        chunk_overlap=50,              # 分块重叠50字符
        top_k=3,                       # 检索前3个最相关文档
        deduplicate=False              # 演示时可关闭去重以加快速度
    )
    rag = RAGChain(config)             # 创建RAG链实例，默认使用内存向量存储

    # 索引示例文档
    print("\n索引示例文档...")          # 提示用户正在索引文档
    texts = [doc.get("text", "") for doc in SAMPLE_DOCUMENTS]      # 提取所有示例文本
    metadatas = [doc.get("metadata", {}) for doc in SAMPLE_DOCUMENTS]  # 提取对应的元数据
    rag.index_documents(texts, metadatas)  # 调用索引方法，处理文档并构建向量存储

    # 单轮问答演示
    print("\n" + "=" * 60)             # 打印分隔线
    print("示例 1：单轮问答")           # 显示演示标题
    questions = [
        "什么是 LangChain？它有什么特点？",   # 第一个问题
        "RAG 系统的工作流程是怎样的？",       # 第二个问题
        "有哪些常见的向量数据库？"            # 第三个问题
    ]
    for q in questions:                         # 遍历每个问题
        result = rag.query(q)                   # 调用查询方法，获取回答
        print(f"\n回答：\n{result['answer']}")  # 打印回答
        print("\n来源：")                       # 打印来源标题
        for src in result['sources']:           # 遍历来源列表
            print(f"   - [{src['index']}] {src['source']}")  # 打印每个来源的索引和来源
        print(f"\n置信度：{result['confidence']:.2f}")       # 打印置信度，保留两位小数
        print("-" * 60)                         # 打印分隔线

    # 多轮对话演示（验证历史维护和指代消解）
    print("\n" + "=" * 60)             # 打印分隔线
    print("示例 2：多轮对话（自动维护历史）")  # 显示演示标题
    q1 = "LangGraph 是什么？"          # 第一个多轮问题
    print(f"\n用户：{q1}")             # 打印用户输入
    result1 = rag.query(q1)            # 执行查询
    print(f"\n助手：{result1['answer']}")  # 打印回答
    q2 = "它的核心概念有哪些？"        # 第二个问题，包含指代“它”
    print(f"\n用户：{q2}")             # 打印用户输入
    result2 = rag.query(q2)            # 执行查询（系统会根据历史改写）
    print(f"\n助手：{result2['answer']}")  # 打印回答
    q3 = "在什么场景下使用它比较合适？"  # 第三个问题，继续指代
    print(f"\n用户：{q3}")             # 打印用户输入
    result3 = rag.query(q3)            # 执行查询
    print(f"\n助手：{result3['answer']}")  # 打印回答

    # 清除历史后开始新对话
    print("\n" + "=" * 60)             # 打印分隔线
    print("示例 3：清除历史后的新对话")  # 显示演示标题
    rag.clear_history()                # 清除对话历史，重置上下文
    q4 = "RAG 是什么？"                # 新对话的第一个问题
    print(f"\n用户：{q4}")             # 打印用户输入
    result4 = rag.query(q4)            # 执行查询
    print(f"\n助手：{result4['answer']}")  # 打印回答
    q5 = "它的工作流程是怎样的？"       # 新对话的第二个问题，此时“它”不再有上文指代，可能需要系统自行判断
    print(f"\n用户：{q5}")             # 打印用户输入
    result5 = rag.query(q5)            # 执行查询
    print(f"\n助手：{result5['answer']}")  # 打印回答

    print("\n" + "=" * 60)             # 打印分隔线
    print("RAG 系统演示完成！")         # 提示演示结束

if __name__ == "__main__":             # 判断是否作为主程序运行
    main()                             # 调用主函数，启动演示