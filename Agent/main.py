"""
多代理智能客服系统 - FastAPI 后端
基于 LangGraph 的意图识别、多专业 Agent 协作、质量检查与人工升级
"""

import logging
import time
import threading
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入您的客服系统核心类
from multi_agent import CustomerServiceSystem

# ==================== 配置加载 ====================
dotenv.load_dotenv()

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 全局状态与并发保护 ====================
service_lock = threading.Lock()
service_system: Optional[CustomerServiceSystem] = None

stats: Dict[str, Any] = {
    "total_queries": 0,
    "total_escalations": 0,
    "last_query_time": None,
    "confidences": [],          # 最近 100 次意图置信度
    "quality_scores": []        # 最近 100 次质量评分
}

# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    global service_system
    logger.info("智能客服系统 API 启动中...")
    try:
        service_system = CustomerServiceSystem()
        logger.info("客服系统实例已创建，所有组件初始化完成")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise
    yield
    logger.info(f"智能客服系统 API 关闭，共处理 {stats['total_queries']} 次查询")

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="多代理智能客服系统 API",
    description="基于 LangGraph 的意图分类、多专业 Agent 协作与质量保障的智能客服",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境请替换为具体域名
    allow_credentials=False,
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
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    use_history: bool = Field(True, description="是否参考对话历史")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="客服回复")
    intent: str = Field(..., description="识别出的用户意图")
    confidence: float = Field(..., ge=0.0, le=1.0, description="意图置信度")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="质检评分")
    escalated: bool = Field(..., description="是否已升级到人工")
    timestamp: str = Field(..., description="响应时间戳")

class HistoryResponse(BaseModel):
    chat_history: List[Dict[str, str]]
    count: int

class HealthResponse(BaseModel):
    status: str
    service_initialized: bool
    total_queries: int
    timestamp: str

class StatsResponse(BaseModel):
    total_queries: int
    total_escalations: int
    average_confidence: float
    average_quality_score: float
    last_query_time: Optional[str]

# ==================== 工具函数 ====================
def get_service() -> CustomerServiceSystem:
    """安全获取客服系统实例，未初始化则返回 503"""
    if service_system is None:
        raise HTTPException(status_code=503, detail="客服系统尚未初始化")
    return service_system

def update_stats(confidence: float, quality_score: float, escalated: bool):
    """线程安全地更新统计信息（调用前需持有 service_lock）"""
    stats["total_queries"] += 1
    if escalated:
        stats["total_escalations"] += 1
    stats["last_query_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    stats["confidences"].append(confidence)
    if len(stats["confidences"]) > 100:
        stats["confidences"].pop(0)
    stats["quality_scores"].append(quality_score)
    if len(stats["quality_scores"]) > 100:
        stats["quality_scores"].pop(0)

# ==================== API 端点 ====================


# 全局变量增加一个用于保护历史操作的锁（也可复用 service_lock，但为了清晰使用独立的）
history_lock = threading.Lock()   # 保护 current_history 的操作

# 修改 /query 端点
@app.post("/query", response_model=QueryResponse, tags=["对话"])
async def query_service(request: QueryRequest):
    """向客服系统提问"""
    srv = get_service()
    
    # 准备对话历史
    with history_lock:
        if request.use_history and hasattr(srv, 'current_history'):
            chat_history = list(srv.current_history)  # 副本，避免后续修改影响
        else:
            chat_history = []
    
    # 执行客服处理（无锁，允许并发）
    try:
        result = srv.handle_message(request.question, chat_history)
    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    # 更新历史（锁保护）
    with history_lock:
        if hasattr(srv, 'current_history'):
            srv.current_history.append({"role": "user", "content": request.question})
            srv.current_history.append({"role": "assistant", "content": result["response"]})
    
    # 更新统计（单独锁）
    with service_lock:
        update_stats(result["confidence"], result["quality_score"], result["escalated"])
    
    return QueryResponse(
        answer=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        quality_score=result["quality_score"],
        escalated=result["escalated"],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

# 修改 /history 端点，添加锁保护
@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """获取当前对话历史"""
    srv = get_service()
    with history_lock:
        if hasattr(srv, 'current_history'):
            history = list(srv.current_history)
        else:
            history = []
    return HistoryResponse(chat_history=history, count=len(history))

# 修改 /delete 端点
@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """清除对话历史"""
    srv = get_service()
    with history_lock:
        if hasattr(srv, 'current_history'):
            srv.current_history.clear()
    return {"message": "对话历史已清除", "success": True}

# 修改 /reset 端点，安全地替换实例并清空统计
@app.post("/reset", tags=["系统"])
async def reset_system():
    """重置整个系统（重新初始化客服系统并清空统计）"""
    global service_system, stats
    new_system = CustomerServiceSystem()
    with service_lock:
        service_system = new_system
    # 重置统计
    with service_lock:
        stats = {
            "total_queries": 0,
            "total_escalations": 0,
            "last_query_time": None,
            "confidences": [],
            "quality_scores": []
        }
    return {"message": "系统已重置", "success": True}

# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )