# 导入 sys 模块，用于修改 Python 解释器的模块搜索路径
import sys
# 导入 os 模块，用于处理操作系统相关功能，如读取环境变量、文件路径操作
import os
# 导入 logging 模块，用于记录程序运行时的日志信息
import logging
# 导入 time 模块，用于获取当前时间戳、格式化时间以及计算代码执行耗时
import time
# 导入 threading 模块，用于创建线程锁，保护多线程环境下的共享数据
import threading
# 导入 asyncio 模块，用于编写异步代码，并在异步环境中执行同步的阻塞操作
import asyncio
# 从 typing 模块导入常用的类型注解，提高代码可读性和类型安全性
from typing import List, Dict, Optional, Any
# 从 contextlib 导入 asynccontextmanager，用于创建异步上下文管理器，管理资源的启动和关闭
from contextlib import asynccontextmanager

# 将当前文件的父目录（项目根目录）添加到 Python 的模块搜索路径中，以便导入 shared 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从 shared 模块中导入 setup_logging 函数，用于配置日志记录器
from shared import setup_logging

# 导入 uvicorn 服务器，用于运行 FastAPI 应用
import uvicorn
# 从 fastapi 导入核心类 FastAPI（应用实例）和 HTTPException（HTTP 异常）、Request（请求对象）
from fastapi import FastAPI, HTTPException, Request
# 从 fastapi 的中间件模块导入 CORSMiddleware，用于处理跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 从 pydantic 导入 BaseModel 和 Field，用于定义请求和响应的数据模型，并进行字段验证
from pydantic import BaseModel, Field

# 从 rag 模块导入 RAGChain（RAG 核心链）、RAGConfig（配置类）和 SAMPLE_DOCUMENTS（示例文档）
from rag import RAGChain, RAGConfig, SAMPLE_DOCUMENTS

# ==================== 配置加载 ====================
# 导入 dotenv 库并立即调用 load_dotenv()，从 .env 文件加载环境变量到系统环境中
import dotenv; dotenv.load_dotenv()
# 从环境变量中读取 CHROMA_PERSIST_DIRECTORY 的值，如果未设置则默认为 None
PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", None)

# ==================== 日志配置 ====================
# 为当前模块初始化日志记录器，后续可使用 logger.info() 等方法记录日志
logger = setup_logging(__name__)

# ==================== 全局状态与并发保护 ====================
# 创建一个线程锁对象，用于保护全局 RAG 实例和统计数据，确保多线程环境下的线程安全
rag_lock = threading.Lock()
# 全局 RAG 系统实例，初始化为 None，将在应用启动时创建
rag_system: Optional[RAGChain] = None
# 创建一个默认的 RAG 配置对象，使用默认参数
rag_config = RAGConfig()

# 定义一个字典 stats 用于存储系统的运行统计信息
stats: Dict[str, Any] = {
    "total_queries": 0,         # 总共处理的查询次数
    "total_documents": 0,       # 当前索引的原始文档总数（不是分块数量）
    "last_query_time": None,    # 最后一次查询的时间戳
    "confidences": []           # 列表，保存最近最多 100 条查询的置信度
}

# ==================== 应用生命周期 ====================
# 使用装饰器定义异步上下文管理器，管理 FastAPI 应用的启动和关闭过程
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    # 声明函数内将使用全局变量 rag_system
    global rag_system
    # 记录应用正在启动的日志
    logger.info("RAG 问答系统 API 启动中...")
    try:
        # 在启动时创建 RAGChain 实例，传入配置和持久化目录
        rag_system = RAGChain(config=rag_config, persist_directory=PERSIST_DIRECTORY)
        # 记录实例创建成功的日志
        logger.info("RAG 系统实例已创建（未自动索引文档）")
    except Exception as e:
        # 如果初始化失败，记录错误日志
        logger.error(f"系统初始化失败: {e}")
        # 重新抛出异常，阻止应用启动
        raise
    # yield 关键字将函数分为两部分：yield 之前是启动逻辑，yield 之后是关闭逻辑
    yield
    # 应用关闭时，记录总共处理了多少次查询
    logger.info(f"RAG 问答系统 API 关闭，共处理 {stats['total_queries']} 次查询")

# ==================== 创建 FastAPI 应用 ====================
# 创建 FastAPI 应用实例，设置 API 的标题、描述、版本号等信息
app = FastAPI(
    title="RAG 问答系统 API",
    description="基于 LangChain/LangGraph 的智能检索增强生成系统",
    version="1.0.0",
    lifespan=lifespan,             # 绑定上面定义的生命周期函数
    docs_url="/docs",              # 交互式 Swagger 文档的访问路径
    redoc_url="/redoc",            # ReDoc 文档的访问路径
)

# 为应用添加 CORS（跨域资源共享）中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有来源的跨域请求（生产环境应限制为具体域名）
    allow_credentials=False,      # 不允许携带凭据，当 origins 为 "*" 时此参数必须为 False
    allow_methods=["*"],          # 允许所有 HTTP 方法（GET、POST、DELETE 等）
    allow_headers=["*"],          # 允许所有请求头
)

# 自定义一个 HTTP 中间件，用于记录每个请求的处理日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 记录请求到达的时间
    start_time = time.time()
    # 调用下一个中间件或具体的路由处理函数，获取响应对象
    response = await call_next(request)
    # 计算请求处理的总耗时
    duration = time.time() - start_time
    # 记录请求的方法、路径、响应状态码和耗时
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response

# ==================== Pydantic 模型 ====================
# 定义接收单个文档的数据模型，用于请求体
class DocumentInput(BaseModel):
    text: str = Field(..., min_length=1, description="文档内容")  # 文档文本内容，必填，长度至少为 1
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict, description="文档元数据")  # 可选的元数据字典

# 定义批量索引文档的请求体模型
class IndexRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., min_length=1, description="文档列表")  # 文档列表，至少 1 篇
    collection_name: str = Field("default", description="目标集合名称")  # 要索引到的集合名称，默认为 "default"

# 定义索引操作后的响应模型
class IndexResponse(BaseModel):
    success: bool              # 操作是否成功
    document_count: int        # 索引的原始文档数量
    chunk_count: int           # 文档分割后的块（chunk）数量
    message: str               # 操作结果的描述信息

# 定义查询请求的模型
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")  # 用户问题，1 到 1000 字
    use_history: bool = Field(True, description="是否参考对话历史")  # 是否使用对话历史进行查询改写

# 定义查询响应的模型
class QueryResponse(BaseModel):
    answer: str                                    # 系统生成的回答
    sources: List[Dict]                            # 引用的来源信息列表
    confidence: float = Field(..., ge=0.0, le=1.0) # 回答的置信度，范围 0 到 1
    timestamp: str                                 # 生成回答的时间戳

# 定义返回对话历史的响应模型
class HistoryResponse(BaseModel):
    chat_history: List[Dict[str, str]]  # 对话历史列表，每条包含 role 和 content
    count: int                          # 对话历史的条数

# 定义健康检查的响应模型
class HealthResponse(BaseModel):
    status: str               # 服务状态，如 "healthy" 或 "not initialized"
    rag_initialized: bool     # RAG 系统是否已初始化
    document_count: int       # 当前索引的文档数量
    timestamp: str            # 检查时刻的时间戳

# 定义系统运行统计的响应模型
class StatsResponse(BaseModel):
    total_queries: int              # 总查询次数
    average_confidence: float       # 平均置信度
    last_query_time: Optional[str]  # 最后一次查询的时间（可能为 None）
    document_count: int             # 当前索引的文档数量

# ==================== 工具函数 ====================
def get_rag() -> RAGChain:
    """安全获取 RAG 系统实例，如果未初始化则抛出 503 服务不可用异常"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 系统尚未初始化")
    return rag_system

def update_stats(confidence: float):
    """线程安全地更新全局统计信息，调用此函数前需要先获得 rag_lock 锁"""
    stats["total_queries"] += 1                                      # 查询总数加一
    stats["last_query_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")   # 更新最后一次查询时间
    stats["confidences"].append(confidence)                          # 将本次置信度加入列表
    if len(stats["confidences"]) > 100:                              # 如果置信度列表超过 100 条
        stats["confidences"].pop(0)                                  # 移除最早的一条，保持列表大小

# ==================== API 端点 ====================

# 健康检查接口，GET /health
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """系统健康检查"""
    init = rag_system is not None                                     # 检查 RAG 系统是否已初始化
    docs = stats["total_documents"] if init else 0                    # 已初始化则获取文档数，否则为 0
    return HealthResponse(
        status="healthy" if init else "not initialized",              # 根据初始化状态返回 health 或 not initialized
        rag_initialized=init,
        document_count=docs,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")                 # 当前时间戳
    )

# 运行统计接口，GET /stats
@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取运行统计信息"""
    with rag_lock:                                                    # 获得锁，保证线程安全
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.0  # 计算平均置信度
        return StatsResponse(
            total_queries=stats["total_queries"],
            average_confidence=round(avg_conf, 2),
            last_query_time=stats["last_query_time"],
            document_count=stats["total_documents"]
        )

# 索引文档接口，POST /index
@app.post("/index", response_model=IndexResponse, tags=["文档管理"])
async def index_documents(request: IndexRequest):
    """
    批量索引新文档，会重建指定集合（覆盖原有同名集合）
    """
    global stats                                                      # 声明将修改全局 stats 字典
    with rag_lock:                                                    # 获取锁
        r = get_rag()                                                 # 获取 RAG 实例
        try:
            texts = [doc.text for doc in request.documents]            # 提取所有文档的文本
            metadatas = [doc.metadata for doc in request.documents]    # 提取对应的元数据
            r.index_documents(texts, metadatas, collection_name=request.collection_name)  # 执行索引操作
            stats["total_documents"] = len(request.documents)         # 更新全局文档计数为本次索引的文档数
            return IndexResponse(
                success=True,
                document_count=len(request.documents),
                chunk_count=r.document_count,                         # r.document_count 是分块后的块数量
                message=f"成功索引 {len(request.documents)} 个文档到集合 '{request.collection_name}'"
            )
        except Exception as e:
            logger.error(f"索引失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 加载示例文档接口，POST /index/sample
@app.post("/index/sample", response_model=IndexResponse, tags=["文档管理"])
async def index_sample_documents():
    """加载内置的示例文档，替换默认集合的内容"""
    global stats
    with rag_lock:
        r = get_rag()
        try:
            r.clear_history()                                          # 清空对话历史
            texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]          # 提取示例文档的文本
            metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]  # 提取示例文档的元数据
            r.index_documents(texts, metadatas, collection_name="default")  # 索引到默认集合
            stats["total_documents"] = len(SAMPLE_DOCUMENTS)          # 更新文档计数
            return IndexResponse(
                success=True,
                document_count=len(SAMPLE_DOCUMENTS),
                chunk_count=r.document_count,
                message="示例文档索引完成"
            )
        except Exception as e:
            logger.error(f"索引示例文档失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 增量添加文档接口，POST /documents/add
@app.post("/documents/add", response_model=IndexResponse, tags=["文档管理"])
async def add_documents(request: IndexRequest):
    """向当前索引增量添加文档，不会重建索引"""
    global stats
    with rag_lock:
        r = get_rag()
        if not r.retriever:                                            # 如果没有检索器，说明索引尚未创建
            raise HTTPException(status_code=400, detail="尚未创建索引，请先调用 /index")
        try:
            texts = [doc.text for doc in request.documents]
            metadatas = [doc.metadata for doc in request.documents]
            r.add_documents(texts, metadatas)                          # 增量添加文档
            stats["total_documents"] += len(request.documents)         # 累加原始文档数
            return IndexResponse(
                success=True,
                document_count=len(request.documents),
                chunk_count=r.document_count,
                message=f"已添加 {len(request.documents)} 个文档"
            )
        except Exception as e:
            logger.error(f"增量添加失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 列出所有集合接口，GET /collections
@app.get("/collections", response_model=List[str], tags=["集合管理"])
async def list_collections():
    """列出所有持久化集合名称，内存模式下只返回 'default'"""
    with rag_lock:
        r = get_rag()
        if r.persist_directory:                                       # 如果配置了持久化目录
            return r.list_collections()                               # 调用底层方法列出所有集合
        else:
            return ["default"]                                        # 内存模式，直接返回默认集合名

# 删除集合接口，DELETE /collections/{collection_name}
@app.delete("/collections/{collection_name}", tags=["集合管理"])
async def delete_collection(collection_name: str):
    """删除指定的持久化集合，内存模式下不支持"""
    with rag_lock:
        r = get_rag()
        if not r.persist_directory:                                   # 内存模式无法删除
            raise HTTPException(status_code=400, detail="内存模式下不支持删除集合")
        try:
            r.delete_collection(collection_name)                      # 执行删除操作
            stats["total_documents"] = 0                              # 由于无法精确统计剩余文档数，简单置零
            return {"message": f"集合 '{collection_name}' 已删除", "success": True}
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# 问答查询接口，POST /query
@app.post("/query", response_model=QueryResponse, tags=["问答"])
async def query_rag(request: QueryRequest):
    """向 RAG 系统提问，返回回答、来源和置信度"""
    with rag_lock:
        r = get_rag()
        original_history = None
        if not request.use_history:                                    # 如果不需要使用对话历史
            original_history = r.chat_history.copy()                   # 备份当前历史
            r.clear_history()                                          # 清空历史
        try:
            # 在异步环境中，将同步的 query 方法放到线程池中执行，避免阻塞事件循环
            result = await asyncio.to_thread(r.query, request.question)
            update_stats(result["confidence"])                         # 更新统计信息
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                confidence=result["confidence"],
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # 确保在查询结束后（无论成功或失败）恢复对话历史
            if original_history is not None:
                r.chat_history = original_history

# 获取对话历史接口，GET /history
@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """获取当前完整的对话历史"""
    with rag_lock:
        r = get_rag()
        return HistoryResponse(
            chat_history=r.chat_history,
            count=len(r.chat_history)
        )

# 清除对话历史接口，DELETE /history
@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """清除所有对话历史"""
    with rag_lock:
        r = get_rag()
        r.clear_history()                                              # 调用 RAGChain 的 clear_history 方法
        return {"message": "对话历史已清除", "success": True}

# 重置系统接口，POST /reset
@app.post("/reset", tags=["系统"])
async def reset_system():
    """重置整个系统，重新创建 RAG 实例并清空所有统计数据"""
    global rag_system, stats
    with rag_lock:
        rag_system = RAGChain(config=rag_config, persist_directory=PERSIST_DIRECTORY)  # 新建一个 RAGChain 实例
        stats = {                                                                    # 重置统计数据为初始值
            "total_queries": 0,
            "total_documents": 0,
            "last_query_time": None,
            "confidences": []
        }
        return {"message": "系统已重置", "success": True}

# ==================== 启动入口 ====================
# 判断当前脚本是否作为主程序直接运行（而不是被其他模块导入）
if __name__ == "__main__":
    # 启动 uvicorn 服务器，监听所有网络接口的 8002 端口
    uvicorn.run(
        "main:app",               # 第一个参数：模块路径 "main" 中的 "app" 实例
        host="0.0.0.0",           # 绑定到所有 IP 地址，允许外部访问
        port=8002,                # 监听 8002 端口
        reload=True,              # 开发模式下启用热重载，代码修改后自动重启服务
        log_level="info"          # 日志级别设为 info，输出常规信息
    )