"""
RAG 问答系统 - FastAPI 后端
重新设计版本：更合理的端点、并发保护、清晰的配置管理
改进版：修正 CORS 警告、安全的历史恢复、健壮的异常处理
"""

import sys, os, logging, time, threading, asyncio
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import setup_logging

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入 RAG 系统核心类（请确保 rag_system 模块已实现）
from rag import RAGChain, RAGConfig, SAMPLE_DOCUMENTS

# ==================== 配置加载 ====================
import dotenv; dotenv.load_dotenv()
PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", None)

# ==================== 日志配置 ====================
logger = setup_logging(__name__)

# ==================== 全局状态与并发保护 ====================
rag_lock = threading.Lock()
rag_system: Optional[RAGChain] = None
rag_config = RAGConfig()

stats: Dict[str, Any] = {
    "total_queries": 0,
    "total_documents": 0,
    "last_query_time": None,
    "confidences": []          # 最多保存 100 条置信度数据
}

# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    global rag_system
    logger.info("RAG 问答系统 API 启动中...")
    try:
        rag_system = RAGChain(config=rag_config, persist_directory=PERSIST_DIRECTORY)
        logger.info("RAG 系统实例已创建（未自动索引文档）")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise
    yield
    logger.info(f"RAG 问答系统 API 关闭，共处理 {stats['total_queries']} 次查询")

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="RAG 问答系统 API",
    description="基于 LangChain/LangGraph 的智能检索增强生成系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件（已修正：allow_credentials=False 避免与 allow_origins=["*"] 冲突）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境应替换为具体域名
    allow_credentials=False,      # 如需携带 Cookie，请将 allow_origins 改为具体域名并设置 True
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response

# ==================== Pydantic 模型 ====================
class DocumentInput(BaseModel):
    text: str = Field(..., min_length=1, description="文档内容")
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict, description="文档元数据")

class IndexRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., min_length=1, description="文档列表")
    collection_name: str = Field("default", description="目标集合名称")

class IndexResponse(BaseModel):
    success: bool
    document_count: int
    chunk_count: int
    message: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    use_history: bool = Field(True, description="是否参考对话历史")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: str

class HistoryResponse(BaseModel):
    chat_history: List[Dict[str, str]]
    count: int

class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
    document_count: int
    timestamp: str

class StatsResponse(BaseModel):
    total_queries: int
    average_confidence: float
    last_query_time: Optional[str]
    document_count: int

# ==================== 工具函数 ====================
def get_rag() -> RAGChain:
    """安全获取 RAG 系统实例，未初始化则返回 503"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 系统尚未初始化")
    return rag_system

def update_stats(confidence: float):
    """线程安全地更新统计信息（调用前需持有 rag_lock）"""
    stats["total_queries"] += 1
    stats["last_query_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    stats["confidences"].append(confidence)
    if len(stats["confidences"]) > 100:
        stats["confidences"].pop(0)

# ==================== API 端点 ====================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """系统健康检查"""
    init = rag_system is not None
    docs = stats["total_documents"] if init else 0
    return HealthResponse(
        status="healthy" if init else "not initialized",
        rag_initialized=init,
        document_count=docs,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取运行统计信息"""
    with rag_lock:
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.0
        return StatsResponse(
            total_queries=stats["total_queries"],
            average_confidence=round(avg_conf, 2),
            last_query_time=stats["last_query_time"],
            document_count=stats["total_documents"]
        )

@app.post("/index", response_model=IndexResponse, tags=["文档管理"])
async def index_documents(request: IndexRequest):
    """
    批量索引新文档（重建指定集合，会覆盖原有同名集合内容）
    """
    global stats
    with rag_lock:
        r = get_rag()
        try:
            texts = [doc.text for doc in request.documents]
            metadatas = [doc.metadata for doc in request.documents]
            r.index_documents(texts, metadatas, collection_name=request.collection_name)
            stats["total_documents"] = r.document_count  # 同步文档总数
            return IndexResponse(
                success=True,
                document_count=len(request.documents),
                chunk_count=r.document_count,
                message=f"成功索引 {len(request.documents)} 个文档到集合 '{request.collection_name}'"
            )
        except Exception as e:
            logger.error(f"索引失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/sample", response_model=IndexResponse, tags=["文档管理"])
async def index_sample_documents():
    """加载内置示例文档（替换默认集合内容）"""
    global stats
    with rag_lock:
        r = get_rag()
        try:
            # 将 clear_history 放入 try 块，避免异常时影响后续操作
            r.clear_history()
            texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
            metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]
            r.index_documents(texts, metadatas, collection_name="default")
            stats["total_documents"] = r.document_count
            return IndexResponse(
                success=True,
                document_count=len(SAMPLE_DOCUMENTS),
                chunk_count=r.document_count,
                message="示例文档索引完成"
            )
        except Exception as e:
            logger.error(f"索引示例文档失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/add", response_model=IndexResponse, tags=["文档管理"])
async def add_documents(request: IndexRequest):
    """向当前索引增量添加文档（不重建索引）"""
    global stats
    with rag_lock:
        r = get_rag()
        if not r.retriever:
            raise HTTPException(status_code=400, detail="尚未创建索引，请先调用 /index")
        try:
            texts = [doc.text for doc in request.documents]
            metadatas = [doc.metadata for doc in request.documents]
            r.add_documents(texts, metadatas)
            stats["total_documents"] = r.document_count
            return IndexResponse(
                success=True,
                document_count=len(request.documents),
                chunk_count=r.document_count,
                message=f"已添加 {len(request.documents)} 个文档"
            )
        except Exception as e:
            logger.error(f"增量添加失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections", response_model=List[str], tags=["集合管理"])
async def list_collections():
    """列出所有持久化集合名称（内存模式下仅返回 'default'）"""
    with rag_lock:
        r = get_rag()
        if r.persist_directory:
            return r.list_collections()
        else:
            return ["default"]

@app.delete("/collections/{collection_name}", tags=["集合管理"])
async def delete_collection(collection_name: str):
    """删除指定持久化集合（内存模式下不支持）"""
    with rag_lock:
        r = get_rag()
        if not r.persist_directory:
            raise HTTPException(status_code=400, detail="内存模式下不支持删除集合")
        try:
            r.delete_collection(collection_name)
            # 简化处理：清零文档计数（实际应减去被删除集合的文档数）
            stats["total_documents"] = 0
            return {"message": f"集合 '{collection_name}' 已删除", "success": True}
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["问答"])
async def query_rag(request: QueryRequest):
    """向 RAG 系统提问"""
    with rag_lock:
        r = get_rag()
        original_history = None
        # 如果不使用历史，备份并清空
        if not request.use_history:
            original_history = r.chat_history.copy()
            r.clear_history()
        try:
            result = await asyncio.to_thread(r.query, request.question)
            update_stats(result["confidence"])
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
            # 确保无论成功或失败，历史都能恢复
            if original_history is not None:
                r.chat_history = original_history

@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """获取当前对话历史"""
    with rag_lock:
        r = get_rag()
        return HistoryResponse(
            chat_history=r.chat_history,
            count=len(r.chat_history)
        )

@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """清除对话历史"""
    with rag_lock:
        r = get_rag()
        r.clear_history()
        return {"message": "对话历史已清除", "success": True}

@app.post("/reset", tags=["系统"])
async def reset_system():
    """重置整个系统（清空所有数据与统计）"""
    global rag_system, stats
    with rag_lock:
        rag_system = RAGChain(config=rag_config, persist_directory=PERSIST_DIRECTORY)
        stats = {
            "total_queries": 0,
            "total_documents": 0,
            "last_query_time": None,
            "confidences": []
        }
        return {"message": "系统已重置", "success": True}

# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )