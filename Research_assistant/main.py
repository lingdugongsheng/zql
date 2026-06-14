"""
智能研究助手 - FastAPI 后端
基于 LangGraph 的多阶段研究流程：规划→收集→分析→综合→报告→质检（可迭代）
"""
import sys, os, logging, time, asyncio
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import setup_logging

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入研究助手核心函数
from research_assistant import run_research

# ==================== 配置加载 ====================
import dotenv; dotenv.load_dotenv()

# ==================== 日志配置 ====================
logger = setup_logging(__name__)

# ==================== 全局状态 ====================
stats: Dict[str, Any] = {
    "total_research_tasks": 0,
    "last_task_time": None,
    "quality_scores": []        # 最近 100 次质量评分
}

# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    logger.info("智能研究助手 API 启动中...")
    # 可在此处进行模型预热或检查
    yield
    logger.info(f"智能研究助手 API 关闭，共完成 {stats['total_research_tasks']} 个研究任务")

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="智能研究助手 API",
    description="基于 LangGraph 的自动研究报告生成系统，支持多阶段迭代优化",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件（已修正 allow_credentials=False）
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
class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="研究主题")

class CitationModel(BaseModel):
    id: str
    authors: List[str]
    title: str
    source: str
    year: int
    url: Optional[str] = None

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

class HealthResponse(BaseModel):
    status: str
    total_tasks: int
    timestamp: str

class StatsResponse(BaseModel):
    total_tasks: int
    average_quality: float
    last_task_time: Optional[str]

# ==================== 工具函数 ====================
def update_stats(quality_score: float):
    """更新统计信息"""
    stats["total_research_tasks"] += 1
    stats["last_task_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    stats["quality_scores"].append(quality_score)
    if len(stats["quality_scores"]) > 100:
        stats["quality_scores"].pop(0)

# ==================== API 端点 ====================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """系统健康检查"""
    return HealthResponse(
        status="healthy",
        total_tasks=stats["total_research_tasks"],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取运行统计信息"""
    avg_quality = (sum(stats["quality_scores"]) / len(stats["quality_scores"])
                   if stats["quality_scores"] else 0.0)
    return StatsResponse(
        total_tasks=stats["total_research_tasks"],
        average_quality=round(avg_quality, 2),
        last_task_time=stats["last_task_time"]
    )

@app.post("/research", response_model=ResearchResponse, tags=["研究"])
async def create_research(request: ResearchRequest):
    """
    启动一项新的研究任务，返回完整研究报告及相关元数据。
    注意：此操作可能耗时较长（取决于主题复杂度），请合理设置前端超时。
    """
    logger.info(f"收到研究请求: {request.topic}")
    try:
        # 调用研究助手的核心函数，返回结果字典
        result = await asyncio.to_thread(run_research, request.topic)
        if result is None:
            raise HTTPException(status_code=500, detail="研究任务执行失败，请稍后重试。")

        # 提取需要的数据
        report_text = result.get("final_report", "")
        citations = result.get("citations", [])
        quality_score = result.get("quality_score", 0.0)
        iteration_count = result.get("iteration_count", 0)
        search_count = len(result.get("search_results", []))
        analyzed_count = len(result.get("analyzed_sources", []))

        # 更新统计
        update_stats(quality_score)

        # 构造响应
        return ResearchResponse(
            topic=request.topic,
            title=result.get("outline", {}).get("title", request.topic),
            report=report_text,
            citations=[CitationModel(**c) for c in citations],
            quality_score=quality_score,
            iteration_count=iteration_count,
            search_results_count=search_count,
            analyzed_sources_count=analyzed_count,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
    except Exception as e:
        logger.error(f"研究任务异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )