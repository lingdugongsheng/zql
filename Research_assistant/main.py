# 智能研究助手 - FastAPI 后端
# 功能：提供 REST API，接收研究主题，调用 LangGraph 多阶段研究流程，返回 Markdown 报告
# 技术栈：FastAPI + LangGraph + Pydantic + Uvicorn

# 导入系统相关模块，用于路径操作
import sys
import os
# 导入时间模块，用于时间戳和耗时计算
import time
# 导入异步 I/O 模块，用于在异步环境中执行同步的阻塞操作
import asyncio
# 导入线程模块，用于创建锁以保护共享数据
import threading
# 导入类型注解相关的泛型类型
from typing import List, Dict, Optional, Any
# 导入异步上下文管理器，用于定义 FastAPI 应用的生命周期事件
from contextlib import asynccontextmanager

# 将项目根目录添加到 Python 模块搜索路径，以便导入 shared 和 research_assistant 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从 shared 模块导入日志配置函数
from shared import setup_logging

# 导入 uvicorn 服务器，用于运行 FastAPI 应用
import uvicorn
# 从 FastAPI 导入核心类 FastAPI，以及 HTTP 异常和请求对象
from fastapi import FastAPI, HTTPException, Request
# 导入 CORS 中间件，处理跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入 Pydantic 的 BaseModel 和 Field，用于定义数据模型和字段校验
from pydantic import BaseModel, Field

# 从 research_assistant 模块导入核心的运行研究函数
from research_assistant import run_research

# ==================== 配置加载 ====================
# 导入 dotenv 并立即调用 load_dotenv，从 .env 文件中加载环境变量
import dotenv; dotenv.load_dotenv()

# ==================== 日志配置 ====================
# 为当前模块初始化日志记录器
logger = setup_logging(__name__)

# ==================== 全局状态与线程锁 ====================
# 创建一个线程锁，用于保护统计数据的安全访问
stats_lock = threading.Lock()
# 定义全局统计信息字典，记录总任务数、最后一次任务时间和质量评分列表
stats: Dict[str, Any] = {
    "total_research_tasks": 0,   # 累计完成的研究任务数量
    "last_task_time": None,      # 最后一个任务完成的时间戳
    "quality_scores": []         # 最近 100 次质量评分，用于计算平均值
}

# ==================== 应用生命周期 ====================
# 定义异步上下文管理器，处理应用的启动和关闭逻辑
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    # 应用启动时记录日志
    logger.info("智能研究助手 API 启动中...")
    # yield 之前的代码在启动时执行，yield 之后的代码在关闭时执行
    yield
    # 应用关闭时记录处理了多少个研究任务
    logger.info(f"智能研究助手 API 关闭，共完成 {stats['total_research_tasks']} 个研究任务")

# ==================== 创建 FastAPI 应用 ====================
# 创建 FastAPI 应用实例，配置标题、描述、版本和生命周期函数
app = FastAPI(
    title="智能研究助手 API",
    description="基于 LangGraph 的自动研究报告生成系统，支持多阶段迭代优化",
    version="1.0.0",
    lifespan=lifespan,           # 绑定上面定义的生命周期管理
    docs_url="/docs",            # Swagger UI 文档路径
    redoc_url="/redoc",          # ReDoc 文档路径
)

# 添加 CORS 中间件，允许所有来源的跨域请求（开发环境设置，生产应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境请替换为具体域名
    allow_credentials=False,      # 不允许携带 Cookie 等凭据
    allow_methods=["*"],          # 允许所有 HTTP 方法
    allow_headers=["*"],          # 允许所有请求头
)

# 自定义一个 HTTP 中间件，用于记录每个请求的日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 记录请求开始时间，用于计算处理耗时
    start_time = time.time()
    # 调用下一个处理程序，获取响应
    response = await call_next(request)
    # 计算耗时
    duration = time.time() - start_time
    # 输出请求方法、路径、状态码和耗时
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    # 返回响应
    return response

# ==================== Pydantic 模型 ====================
# 定义研究请求的数据模型
class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="研究主题")

# 定义单条参考文献的数据模型
class CitationModel(BaseModel):
    id: str                     # 引用编号，如 "[1]"
    authors: List[str]          # 作者列表
    title: str                  # 文献标题
    source: str                 # 文献来源（期刊/网站等）
    year: int                   # 出版年份
    url: Optional[str] = None   # 可选的 URL 链接

# 定义研究响应的数据模型，包含完整的报告和元数据
class ResearchResponse(BaseModel):
    topic: str = Field(..., description="原始研究主题")
    title: str = Field(..., description="报告标题")
    report: str = Field(..., description="完整 Markdown 报告")
    citations: List[CitationModel] = Field(..., description="参考文献列表")
    quality_score: float = Field(..., description="最终质量评分 (0-10)")
    iteration_count: int = Field(..., description="迭代次数")
    search_results_count: int = Field(..., description="收集的原始资料数")
    analyzed_sources_count: int = Field(..., description="分析后的来源数")
    timestamp: str = Field(..., description="生成时间戳")

# 定义健康检查响应的数据模型
class HealthResponse(BaseModel):
    status: str                 # 服务状态，如 "healthy"
    total_tasks: int            # 已完成的总任务数
    timestamp: str              # 检查时间戳

# 定义统计信息响应的数据模型
class StatsResponse(BaseModel):
    total_tasks: int            # 总任务数
    average_quality: float      # 平均质量评分
    last_task_time: Optional[str]  # 最后一次任务完成时间

# ==================== 工具函数 ====================
def update_stats(quality_score: float):
    """
    线程安全地更新全局统计信息。
    调用此函数前必须已经持有 stats_lock 锁。
    """
    # 总任务计数加一
    stats["total_research_tasks"] += 1
    # 更新最后任务时间
    stats["last_task_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # 添加本次的质量评分
    stats["quality_scores"].append(quality_score)
    # 如果列表长度超过 100，删除最早的一条
    if len(stats["quality_scores"]) > 100:
        stats["quality_scores"].pop(0)

# ==================== API 端点 ====================

# 健康检查端点，GET /health
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """系统健康检查：返回服务状态和已处理的任务总数"""
    # 使用锁保护读取统计数据
    with stats_lock:
        total = stats["total_research_tasks"]
    # 构建并返回健康检查响应
    return HealthResponse(
        status="healthy",
        total_tasks=total,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

# 统计信息端点，GET /stats
@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取运行统计信息：总任务数、平均质量评分和最后一次任务时间"""
    # 加锁读取统计数据
    with stats_lock:
        total = stats["total_research_tasks"]
        last = stats["last_task_time"]
        # 计算平均质量评分，避免除零
        if stats["quality_scores"]:
            avg_quality = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        else:
            avg_quality = 0.0
    # 返回统计信息
    return StatsResponse(
        total_tasks=total,
        average_quality=round(avg_quality, 2),   # 四舍五入保留两位小数
        last_task_time=last
    )

# 核心研究端点，POST /research
@app.post("/research", response_model=ResearchResponse, tags=["研究"])
async def create_research(request: ResearchRequest):
    """
    启动一项新的研究任务，执行多阶段研究流程，返回完整研究报告及相关元数据。
    注意：此操作可能耗时较长（取决于主题复杂度和 LLM 响应速度），前端应设置足够的超时时间。
    """
    # 记录收到的研究主题
    logger.info(f"收到研究请求: {request.topic}")
    try:
        # 将同步的 run_research 函数放到线程池中执行，避免阻塞事件循环
        # asyncio.to_thread 会在单独的线程中运行该函数并返回一个可等待的协程
        result = await asyncio.to_thread(run_research, request.topic)
        # 如果 run_research 返回 None，表示执行失败
        if result is None:
            raise HTTPException(status_code=500, detail="研究任务执行失败，请稍后重试。")

        # 从返回的结果字典中提取各项数据
        report_text = result.get("final_report", "")
        citations = result.get("citations", [])
        quality_score = result.get("quality_score", 0.0)
        iteration_count = result.get("iteration_count", 0)
        search_count = len(result.get("search_results", []))
        analyzed_count = len(result.get("analyzed_sources", []))

        # 更新全局统计数据（加锁）
        with stats_lock:
            update_stats(quality_score)

        # 构建并返回 ResearchResponse，将 citations 列表转换为 CitationModel 对象列表
        return ResearchResponse(
            topic=request.topic,
            title=result.get("outline", {}).get("title", request.topic),  # 从大纲中提取标题，若无则用主题代替
            report=report_text,
            citations=[CitationModel(**c) for c in citations],  # 字典解包构造 CitationModel
            quality_score=quality_score,
            iteration_count=iteration_count,
            search_results_count=search_count,
            analyzed_sources_count=analyzed_count,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
    except Exception as e:
        # 捕获所有异常，记录详细日志并返回 500 错误
        logger.error(f"研究任务异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 启动入口 ====================
# 当此文件直接运行（而非被导入）时，启动 uvicorn 服务器
if __name__ == "__main__":
    uvicorn.run(
        "main:app",               # 应用实例的位置：模块 "main" 中的 "app" 对象
        host="0.0.0.0",           # 监听所有网络接口
        port=8003,                # 端口号 8003
        reload=True,              # 开发模式，代码变更时自动重启
        log_level="info"          # 日志级别为 info
    )