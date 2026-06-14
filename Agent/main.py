# 文件：多代理智能客服系统后端主程序
# 功能：提供 RESTful API，接收用户消息，调用多 Agent 客服系统，返回回复、意图、质量评分等信息
# 技术栈：FastAPI + LangGraph + Pydantic + Uvicorn

# 导入系统模块，用于操作环境变量、文件路径
import sys, os
# 导入日志模块，用于记录运行时的信息、警告和错误
import logging
# 导入时间模块，用于获取当前时间戳和计算请求耗时
import time
# 导入线程模块，用于创建锁，保证多线程环境下对共享资源的安全访问
import threading
# 导入异步IO模块，用于将同步的阻塞操作放到线程池中执行，避免阻塞事件循环
import asyncio
# 导入类型注解，用于声明变量、函数的参数和返回值类型，提高代码可读性
from typing import List, Dict, Optional, Any
# 导入异步上下文管理器装饰器，用于定义 FastAPI 应用的生命周期事件（启动/关闭）
from contextlib import asynccontextmanager

# 将项目根目录（当前文件的上级目录）添加到 Python 模块搜索路径中，以便可以直接导入项目内的其他模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从 shared 包导入 setup_logging 函数，该函数用于配置全局日志格式和级别
from shared import setup_logging

# 导入 uvicorn 服务器，用于运行 FastAPI 应用
import uvicorn
# 导入 FastAPI 核心类、HTTPException 异常类、Request 对象
from fastapi import FastAPI, HTTPException, Request
# 导入 CORS 中间件，用于处理跨域请求，允许前端从不同域名访问后端 API
from fastapi.middleware.cors import CORSMiddleware
# 导入 pydantic 的 BaseModel 和 Field，用于定义请求和响应的数据结构，并自动进行数据校验
from pydantic import BaseModel, Field

# 导入多代理客服系统的核心类（包含意图分类、专业 Agent、质量检查等功能）
from multi_agent import CustomerServiceSystem

# ==================== 配置加载 ====================
# 导入 python-dotenv 库，用于从 .env 文件加载环境变量
import dotenv
# 调用 load_dotenv 函数，读取 .env 文件中的配置并注入到 os.environ 中
dotenv.load_dotenv()

# ==================== 日志配置 ====================
# 使用 shared 中的函数初始化日志记录器，__name__ 会生成类似 "main" 的日志名称，便于区分模块
logger = setup_logging(__name__)

# ==================== 全局状态与并发保护 ====================
# 创建一个线程锁，用于保护全局服务实例和统计信息的并发读写，避免数据竞争
service_lock = threading.Lock()
# 声明一个全局变量，用于保存客服系统的单例实例，初始为空
service_system: Optional[CustomerServiceSystem] = None

# 定义一个全局字典，用于存储系统的运行统计信息，所有 API 端点均可访问和更新
stats: Dict[str, Any] = {
    "total_queries": 0,          # 累计查询总数
    "total_escalations": 0,      # 累计升级到人工的次数
    "last_query_time": None,     # 最后一次查询发生的时间（ISO8601 格式字符串）
    "confidences": [],           # 最近 100 次意图置信度列表
    "quality_scores": []         # 最近 100 次质量评分列表
}

# ==================== 应用生命周期 ====================
# 使用 asynccontextmanager 装饰器定义一个异步上下文管理器，用于处理应用的启动和关闭逻辑
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    # 声明要修改全局变量 service_system
    global service_system
    # 在启动时记录日志
    logger.info("智能客服系统 API 启动中...")
    try:
        # 实例化客服系统，这会触发模型加载、Agent 初始化等操作
        service_system = CustomerServiceSystem()
        logger.info("客服系统实例已创建，所有组件初始化完成")
    except Exception as e:
        # 如果初始化失败，记录错误并重新抛出异常，阻止应用继续启动
        logger.error(f"系统初始化失败: {e}")
        raise
    # yield 之前的部分在启动时执行，yield 之后的部分在应用关闭时执行
    yield
    # 应用关闭时，打印总处理查询数
    logger.info(f"智能客服系统 API 关闭，共处理 {stats['total_queries']} 次查询")

# ==================== 创建 FastAPI 应用 ====================
# 创建 FastAPI 应用实例，并设置元数据
app = FastAPI(
    title="多代理智能客服系统 API",
    description="基于 LangGraph 的意图分类、多专业 Agent 协作与质量保障的智能客服",
    version="1.0.0",
    lifespan=lifespan,   # 将上面定义的生命周期函数绑定到应用
    docs_url="/docs",    # 设置 Swagger UI 文档的访问路径
    redoc_url="/redoc",  # 设置 ReDoc 文档的访问路径
)

# 添加 CORS 中间件，解决浏览器跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有来源的请求（生产环境应替换为实际的前端域名）
    allow_credentials=False,      # 不允许在跨域请求中携带 Cookie 等凭据（当 origins 为 "*" 时，此值必须为 False）
    allow_methods=["*"],          # 允许所有 HTTP 方法（GET, POST, PUT, DELETE 等）
    allow_headers=["*"],          # 允许所有请求头
)

# 添加一个全局的 HTTP 中间件，用于记录每一个请求的日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个 HTTP 请求的方法、路径、状态码和耗时"""
    # 记录当前时间，用于计算处理耗时
    start_time = time.time()
    # 调用下一个中间件或路由处理函数，并等待响应
    response = await call_next(request)
    # 计算从请求到响应返回的总时间
    duration = time.time() - start_time
    # 输出格式化的日志信息
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    # 将响应返回给客户端
    return response

# ==================== Pydantic 模型 ====================
# 这些模型定义了 API 期望接收和返回的数据结构，FastAPI 会自动进行校验和序列化

class QueryRequest(BaseModel):
    """查询请求的数据模型"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    # 是否在本次查询中考虑对话历史（用于多轮对话）
    use_history: bool = Field(True, description="是否参考对话历史")

class QueryResponse(BaseModel):
    """查询响应的数据模型"""
    answer: str = Field(..., description="客服回复")                 # 最终的回复文本
    intent: str = Field(..., description="识别出的用户意图")        # 例如 tech_support, order_service
    confidence: float = Field(..., ge=0.0, le=1.0, description="意图置信度")  # 范围 0~1
    quality_score: float = Field(..., ge=0.0, le=1.0, description="质检评分")  # 回复质量评分
    escalated: bool = Field(..., description="是否已升级到人工")   # 是否建议转接人工
    timestamp: str = Field(..., description="响应时间戳")           # 生成此响应的时间

class HistoryResponse(BaseModel):
    """对话历史响应的数据模型"""
    chat_history: List[Dict[str, str]]  # 对话历史列表，每条记录是 {"role": "user/assistant", "content": "..."}
    count: int                         # 历史记录的总条数

class HealthResponse(BaseModel):
    """健康检查响应的数据模型"""
    status: str                    # 服务状态，例如 "healthy" 或 "not initialized"
    service_initialized: bool      # 服务是否已成功初始化
    total_queries: int             # 自启动以来处理的总查询次数
    timestamp: str                 # 生成此响应的时间

class StatsResponse(BaseModel):
    """统计信息响应的数据模型"""
    total_queries: int             # 总查询次数
    total_escalations: int        # 总升级次数
    average_confidence: float     # 平均置信度（基于最近记录）
    average_quality_score: float  # 平均质量评分
    last_query_time: Optional[str] # 最后一次查询的时间，可能为空

# ==================== 工具函数 ====================
def get_service() -> CustomerServiceSystem:
    """
    安全获取客服系统实例。
    如果实例尚未初始化，抛出 HTTPException 503 错误，避免后续空指针错误。
    """
    if service_system is None:
        raise HTTPException(status_code=503, detail="客服系统尚未初始化")
    return service_system

def update_stats(confidence: float, quality_score: float, escalated: bool):
    """
    线程安全地更新全局统计信息。
    注意：调用此函数前必须已经获取 service_lock 锁。
    将最新的查询指标追加到统计列表中，并维护列表长度不超过 100。
    """
    stats["total_queries"] += 1             # 累计查询次数 +1
    if escalated:
        stats["total_escalations"] += 1     # 如果需要转人工，累计升级次数 +1
    # 更新最后一次查询时间，格式为 "YYYY-MM-DDTHH:MM:SS"
    stats["last_query_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    stats["confidences"].append(confidence)  # 将本次置信度存入列表
    if len(stats["confidences"]) > 100:      # 控制列表大小，仅保留最近 100 条
        stats["confidences"].pop(0)
    stats["quality_scores"].append(quality_score)
    if len(stats["quality_scores"]) > 100:
        stats["quality_scores"].pop(0)

# ==================== API 端点 ====================

# 单独创建一个锁用于保护对话历史的并发读写（避免与 service_lock 混合使用导致性能下降）
history_lock = threading.Lock()

# 1. 健康检查端点
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """系统健康检查：返回服务状态、初始化情况及查询总数"""
    # 判断服务实例是否已创建
    init = service_system is not None
    # 如果已初始化，读取当前累计查询数，否则为 0
    total = stats["total_queries"] if init else 0
    # 构造 HealthResponse 对象返回
    return HealthResponse(
        status="healthy" if init else "not initialized",
        service_initialized=init,
        total_queries=total,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

# 2. 核心查询端点
@app.post("/query", response_model=QueryResponse, tags=["对话"])
async def query_service(request: QueryRequest):
    """向客服系统提问，返回回答、意图、置信度等信息，并更新对话历史与统计"""
    # 获取当前服务实例
    srv = get_service()
    
    # --- 准备对话历史（线程安全地读取）---
    with history_lock:
        # 如果请求允许使用历史，且服务实例中存在 current_history 属性
        if request.use_history and hasattr(srv, 'current_history'):
            # 创建一份副本，防止在处理过程中被其他请求修改
            chat_history = list(srv.current_history)
        else:
            chat_history = []
    
    # --- 执行客服处理（在独立线程中运行，避免阻塞异步事件循环）---
    try:
        # 将同步方法 handle_message 包装到线程池中执行
        result = await asyncio.to_thread(srv.handle_message, request.question, chat_history)
    except Exception as e:
        # 如果处理过程中出现异常，记录错误日志并返回 500 状态码
        logger.error(f"查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    # --- 更新对话历史（线程安全地写入）---
    with history_lock:
        if hasattr(srv, 'current_history'):
            # 将本次的用户问题和客服回复追加到历史列表末尾
            srv.current_history.append({"role": "user", "content": request.question})
            srv.current_history.append({"role": "assistant", "content": result["response"]})
    
    # --- 更新统计信息（需要 service_lock 保护）---
    with service_lock:
        update_stats(result["confidence"], result["quality_score"], result["escalated"])
    
    # --- 返回格式化后的响应 ---
    return QueryResponse(
        answer=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        quality_score=result["quality_score"],
        escalated=result["escalated"],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

# 3. 获取对话历史端点
@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """获取当前对话历史（用于展示或调试）"""
    srv = get_service()
    with history_lock:
        # 安全地获取历史列表的副本
        if hasattr(srv, 'current_history'):
            history = list(srv.current_history)
        else:
            history = []
    # 返回历史记录和总数
    return HistoryResponse(chat_history=history, count=len(history))

# 4. 清除对话历史端点
@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """清除当前会话的所有对话历史，开始新的对话"""
    srv = get_service()
    with history_lock:
        if hasattr(srv, 'current_history'):
            srv.current_history.clear()  # 清空列表
    return {"message": "对话历史已清除", "success": True}

# 5. 重置系统端点
@app.post("/reset", tags=["系统"])
async def reset_system():
    """
    重置整个系统：
    - 重新初始化客服系统实例（丢弃旧实例）
    - 清空所有统计数据
    - 清空对话历史（隐含在新建实例中）
    """
    global service_system, stats
    # 创建一个全新的客服系统实例
    new_system = CustomerServiceSystem()
    # 在 service_lock 保护下替换全局实例，确保并发安全
    with service_lock:
        service_system = new_system
    # 同样在锁保护下重置统计数据
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
# 当当前模块作为主程序运行时（而不是被导入），启动 uvicorn 服务器
if __name__ == "__main__":
    uvicorn.run(
        "main:app",          # 指定应用对象的位置："main" 是模块名，"app" 是 FastAPI 实例
        host="0.0.0.0",     # 监听所有可用的网络接口
        port=8001,           # 监听 8001 端口
        reload=True,         # 开启代码修改后自动重启（开发模式）
        log_level="info"     # 设置日志输出级别为 info
    )