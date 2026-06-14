# %%
# ==================== 第一部分：导入库 ====================
import os, sys, json
import logging                     # 日志模块，记录程序运行信息
from typing import TypedDict, Literal, Annotated, Optional  # 类型注解，提高代码可读性
from datetime import datetime      # 获取当前时间，用于报告生成时间戳
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import safe_parse_json, setup_logging, ModelCache, llm_invoke_with_retry

import dotenv                      # 从 .env 文件加载环境变量（API 密钥等）
from langchain_core.messages import HumanMessage, AIMessage  # 人类和 AI 消息类型
from langgraph.graph import StateGraph, START, END, add_messages  # 状态图核心类及消息列表合并函数
from langgraph.checkpoint.memory import MemorySaver           # 内存检查点保存器，记录状态历史
from pydantic import BaseModel, Field                         # 数据模型，用于结构化输出定义

# %%
# ==================== 第二部分：日志配置 ====================
logger = setup_logging(__name__)

# %%
# ==================== 第三部分：JSON 解析辅助函数 ====================
# safe_parse_json imported from shared.utils

# %%
# ==================== 第四部分：环境变量与模型单例 ====================
dotenv.load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")     # 读取 DeepSeek API 密钥
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")   # 读取 DeepSeek API 基础 URL

_model_cache = ModelCache(temperature=0.3, max_tokens=2000)

if not DEEPSEEK_API_KEY:
    raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置，请在 .env 文件中添加该变量。")
if not DEEPSEEK_BASE_URL:
    raise ValueError("环境变量 DEEPSEEK_BASE_URL 未设置，请在 .env 文件中添加该变量。")

def get_model():
    """Get the shared model instance."""
    return _model_cache.get()

# %%
# ==================== 第五部分：Pydantic 数据模型 ====================
# 定义研究中使用的各种数据结构，用于类型约束和序列化

class SearchResult(BaseModel):
    """搜索结果：模拟从数据库或网络检索到的单条文献/新闻"""
    title: str                                 # 文献/文章标题
    source: str                                # 来源（如期刊名、网站名）
    url: str                                   # 可访问链接
    snippet: str                               # 内容摘要或片段
    relevance_score: float = Field(ge=0.0, le=1.0)  # 与主题的相关性评分（0~1）
    publish_date: Optional[str] = None         # 发布日期（可选）


class ResearchFinding(BaseModel):
    """研究发现：从分析中提炼的关键发现，包含证据和来源"""
    topic: str                                 # 所属研究主题
    key_points: list[str]                      # 核心观点列表
    evidence: list[str]                        # 支持该发现的证据列表
    confidence: float = Field(ge=0.0, le=1.0)  # 对该发现的确信度（0~1）
    sources: list[str]                         # 引用来源的标题或ID列表


class ResearchOutline(BaseModel):
    """研究大纲：LLM 生成的研究计划，指导后续步骤"""
    title: str                                 # 研究报告标题
    abstract: str                              # 研究摘要（简短描述）
    sections: list[str]                        # 报告各章节标题列表，如["引言","方法","结论"]
    key_questions: list[str]                   # 需要回答的关键研究问题
    methodology: str                           # 拟采用的研究方法描述


class Citation(BaseModel):
    """参考文献：符合学术规范的单条引用"""
    id: str                                    # 引用标识符（如"[1]"）
    authors: list[str]                         # 作者列表
    title: str                                 # 文献标题
    source: str                                # 文献出处（期刊/会议/网站名）
    year: int                                  # 发表年份
    url: Optional[str] = None                  # 在线链接（可选）


class ResearchReport(BaseModel):
    """最终研究报告：整合所有阶段产出的结构化报告"""
    title: str                                 # 报告标题
    executive_summary: str                     # 执行摘要（概括核心内容）
    introduction: str                          # 引言部分
    methodology: str                           # 方法论部分
    findings: list[str]                        # 主要发现列表
    analysis: str                              # 分析讨论部分
    conclusions: list[str]                     # 结论列表
    recommendations: list[str]                 # 建议/展望列表
    citations: list[Citation]                  # 参考文献列表，每个元素为 Citation 对象
    generated_at: str                          # 报告生成时间（ISO格式字符串）


# %%
# ==================== 第六部分：研究状态定义 ====================
class ResearchState(TypedDict):
    """研究助手工作流中共享的状态，贯穿所有节点"""
    messages: Annotated[list, add_messages]  # 消息历史，使用 add_messages 合并新消息
    research_topic: str                      # 研究主题
    research_questions: list[str]            # 研究问题列表
    search_results: list[dict]               # 收集到的原始搜索结果
    analyzed_sources: list[dict]             # 分析后的来源摘要
    outline: dict                            # 研究大纲
    findings: list[dict]                     # 关键发现
    draft_sections: dict                     # 各章节草稿（键为章节名，值为内容）
    final_report: str                        # 最终报告文本
    citations: list[dict]                    # 引用列表
    current_phase: str                       # 当前阶段（planning, information_gathering, ...）
    iteration_count: int                     # 迭代次数（用于质量检查后重新收集资料）
    quality_score: float                     # 质量评分 (0-10)
    quality_feedback: str                    # 质量反馈文本

# %%
# ==================== 第七部分：模拟数据源 ====================
# 模拟学术数据库和网络搜索结果，实际应用可替换为真实 API 调用

ACADEMIC_DATABASE = {
    "人工智能": [
        {"title": "深度学习在自然语言处理中的应用综述", "authors": ["张明", "李华"], "source": "计算机学报",
         "year": 2024, "snippet": "本文综述了深度学习技术在NLP领域的最新进展...", "url": "https://example.com/paper1"},
        {"title": "人工智能在医疗影像诊断中的突破", "authors": ["王强", "赵丽"], "source": "医学信息学杂志",
         "year": 2023, "snippet": "研究AI在CT、MRI图像识别中的应用，准确率达95%...", "url": "https://example.com/paper2"},
        {"title": "强化学习在机器人控制中的进展", "authors": ["刘洋", "陈晨"], "source": "自动化学报",
         "year": 2024, "snippet": "探讨强化学习算法在复杂环境下的控制策略...", "url": "https://example.com/paper3"},
    ],
    "气候变化": [
        {"title": "全球变暖对农业生产的影响", "authors": ["Smith J.", "Brown K."], "source": "Nature Climate Change",
         "year": 2023, "snippet": "分析气温上升对全球主要作物产量的影响...", "url": "https://example.com/climate1"},
        {"title": "碳中和政策的经济效应分析", "authors": ["李娜", "周杰"], "source": "经济研究",
         "year": 2024, "snippet": "评估中国碳中和目标对能源结构和经济增长的影响...", "url": "https://example.com/climate2"},
    ],
    "量子计算": [
        {"title": "量子计算在密码学中的应用", "authors": ["Johnson M."], "source": "Quantum Information",
         "year": 2023, "snippet": "讨论量子计算机对现有加密体系的威胁...", "url": "https://example.com/quantum1"},
    ]
}

WEB_SEARCH_RESULTS = {
    "人工智能": [
        {"title": "OpenAI发布GPT-5：AI能力再次飞跃", "source": "科技新闻网", "url": "https://news.example.com/gpt5",
         "snippet": "OpenAI最新模型在推理、创造力和多模态方面取得重大突破...", "date": "2024-12"},
        {"title": "AI辅助诊断系统在多家医院投入试用", "source": "健康报", "url": "https://health.example.com/ai",
         "snippet": "国内多家三甲医院开始试点AI辅助诊断，提高早期疾病检出率...", "date": "2024-11"},
    ],
    "气候变化": [
        {"title": "联合国气候变化大会达成新协议", "source": "环球时报", "url": "https://world.example.com/un",
         "snippet": "各国承诺进一步削减温室气体排放...", "date": "2024-10"},
    ],
    "量子计算": [
        {"title": "IBM发布千量子比特处理器", "source": "量子科技网", "url": "https://quantum.example.com/ibm",
         "snippet": "IBM宣布成功制造超过1000量子比特的处理器...", "date": "2024-11"},
    ]
}

# %%
# ==================== 第八部分：工具函数 ====================
def search_academic_database(topic: str, max_results: int = 5) -> list[dict]:
    """从模拟学术数据库中搜索与主题相关的论文"""
    results = []
    for key, papers in ACADEMIC_DATABASE.items():
        # 双向模糊匹配：主题包含数据库键或数据库键包含主题
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            for paper in papers[:max_results]:                 # 只取前 max_results 篇
                results.append({**paper, "type": "academic", "relevance_score": 0.9})  # 添加类型和评分
    return results[:max_results]                               # 返回最多 max_results 条结果

def search_web(topic: str, max_results: int = 5) -> list[dict]:
    """从模拟网络新闻中搜索与主题相关的文章"""
    results = []
    for key, items in WEB_SEARCH_RESULTS.items():
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            for item in items[:max_results]:
                results.append({**item, "type": "web", "relevance_score": 0.8})
    return results[:max_results]

def format_citation(source: dict, citation_id: str) -> Citation:
    """将原始来源数据格式化为 Citation 对象"""
    return Citation(
        id=citation_id,
        authors=source.get("authors", ["Unknown"]),        # 若缺少作者，默认为 Unknown
        title=source.get("title", "Untitled"),
        source=source.get("source", "Unknown"),
        year=source.get("year", 2024),
        url=source.get("url")                               # url 可选
    )

# ==================== LLM 调用重试机制 ====================
def llm_call_with_retry(prompt_messages, max_retries=3, delay=1.5):
    """Backward-compat wrapper; delegates to shared.llm_invoke_with_retry."""
    return llm_invoke_with_retry(_model_cache, prompt_messages, max_retries, delay)

# %%
# ==================== 第九部分：智能体节点与图构建 ====================
def create_research_assistant():
    """
    创建并返回编译好的研究助手状态图。
    流程：规划 -> 信息收集 -> 分析 -> 综合 -> 报告生成 -> 质量检查
          质量不合格且未超迭代次数时，返回信息收集节点重新搜索更高质量资料。
    """

    # ---- 1. 研究规划节点 ----
    def planning_node(state: ResearchState) -> dict:
        """根据研究主题，让 LLM 生成研究计划（标题、大纲、关键问题等）"""
        logger.info("研究规划阶段...")
        topic = state["research_topic"]               # 获取研究主题

        planning_prompt = f"""你是一位资深研究员。请为以下研究主题制定研究计划。
        研究主题：{topic}
        请严格按照以下JSON格式返回：
        {{
            "title": "研究标题",
            "abstract": "摘要（100字以内）",
            "sections": ["章节1", "章节2", "章节3", "章节4"],
            "key_questions": ["问题1", "问题2", "问题3"],
            "methodology": "研究方法论描述"
        }}
        只返回JSON，不要其他文字。"""

        response = llm_call_with_retry([HumanMessage(content=planning_prompt)])  # 调用 LLM
        outline = safe_parse_json(response.content, {       # 安全解析 JSON
            # 默认大纲，当 LLM 输出无法解析时使用
            "title": f"{topic}研究",
            "abstract": f"本研究探讨{topic}的相关问题。",
            "sections": ["引言", "文献综述", "研究方法", "结果分析", "结论"],
            "key_questions": [f"{topic}的现状如何？", f"{topic}的发展趋势是什么？", f"{topic}面临哪些挑战？"],
            "methodology": "文献研究与案例分析相结合"
        })

        logger.info(f"标题: {outline.get('title')}, 章节数: {len(outline.get('sections', []))}")
        return {
            "outline": outline,
            "research_questions": outline.get("key_questions", []),    # 提取关键研究问题
            "current_phase": "information_gathering",                 # 下一阶段
            "messages": [AIMessage(content=f"研究计划已制定：{outline.get('title')}")]
        }

    # ---- 2. 信息收集节点 ----
    def information_gathering_node(state: ResearchState) -> dict:
        """模拟信息检索：调用学术数据库和网络搜索，收集相关资料"""
        logger.info("信息收集阶段...")
        topic = state["research_topic"]
        academic = search_academic_database(topic)       # 搜索学术文献
        web = search_web(topic)                          # 搜索网络资源
        all_results = academic + web                     # 合并结果
        logger.info(f"收集到 {len(all_results)} 条资料")
        if not all_results:
            logger.warning("未找到相关资料，报告可能缺乏依据")
        return {
            "search_results": all_results,
            "current_phase": "analysis",
            "messages": [AIMessage(content=f"已收集 {len(all_results)} 条相关资料")]
        }

    # ---- 3. 信息分析节点 ----
    def analysis_node(state: ResearchState) -> dict:
        """对收集的资料进行深度分析，提取关键发现、证据和信息缺口"""
        logger.info("信息分析阶段...")
        topic = state["research_topic"]
        search_results = state.get("search_results", [])       # 获取原始资料
        questions = state.get("research_questions", [])        # 获取研究问题
        quality_feedback = state.get("quality_feedback", "")   # 可能的质量反馈

        # 如果有质量反馈（来自之前迭代），将其作为分析提示，指导 LLM 改进
        feedback_section = ""
        if quality_feedback:
            feedback_section = f"\n\n本次为迭代改进，请特别针对以下反馈调整分析：\n{quality_feedback}"

        sources_summary = "\n".join([f"- {r['title']}: {r.get('snippet', '')}" for r in search_results[:8]])  # 最多取8条摘要
        analysis_prompt = f"""基于以下资料，对研究主题进行深入分析：
        研究主题：{topic}
        核心问题：\n""" + "\n".join(f"- {q}" for q in questions) + f"""
        资料：{sources_summary}{feedback_section}

        请提供 JSON 格式输出，包含：
        - key_findings: 列表，每个元素包含 finding, evidence, confidence, sources
        - analysis_points: 列表，观点比较等
        - information_gaps: 列表
        只返回 JSON。"""

        response = llm_call_with_retry([HumanMessage(content=analysis_prompt)])
        analysis_data = safe_parse_json(response.content, {})

        # 处理 LLM 返回的 findings，确保字段完整
        key_findings = analysis_data.get("key_findings", [])
        if not key_findings:  # 如果 LLM 未返回 findings，构造默认条目
            key_findings = [{
                "finding": f"关于{topic}的初步发现",
                "evidence": "综合资料显示",
                "confidence": 0.8,
                "sources": [r["title"] for r in search_results[:3]]
            }]

        findings = []
        for f in key_findings:
            findings.append({
                "topic": topic,
                "key_points": [f.get("finding", "")],
                "evidence": [f.get("evidence", "")] if isinstance(f.get("evidence"), str) else f.get("evidence", []),
                "confidence": f.get("confidence", 0.7),
                "sources": f.get("sources", [])
            })

        # 整理分析后的来源摘要
        analyzed_sources = []
        for i, result in enumerate(search_results[:6]):   # 最多6个来源
            analyzed_sources.append({
                "id": f"src_{i+1}",
                "title": result["title"],
                "key_takeaways": result.get("snippet", "")[:100],   # 截取前100字符
                "relevance": result.get("relevance_score", 0.5)
            })

        logger.info(f"提取了 {len(findings)} 组关键发现")
        return {
            "findings": findings,
            "analyzed_sources": analyzed_sources,
            "current_phase": "synthesis",
            "messages": [AIMessage(content="分析完成")]
        }

    # ---- 4. 知识综合节点（生成各章节草稿） ----
    def synthesis_node(state: ResearchState) -> dict:
        """根据分析结果，为报告的每个章节撰写草稿内容"""
        logger.info("知识综合阶段...")
        topic = state["research_topic"]
        outline = state.get("outline", {})
        findings = state.get("findings", [])
        sources = state.get("analyzed_sources", [])

        sections = outline.get("sections", ["引言", "方法", "发现", "结论"])  # 获取章节列表
        source_list = "\n".join([f"- {s['title']}" for s in sources[:5]])    # 最多5个来源

        synthesis_prompt = f"""你是专业报告撰写人。请根据以下信息，为研究报告的每个章节撰写内容（每章150-300字）。

        研究主题：{topic}
        大纲摘要：{outline.get('abstract', '')}
        关键发现：{json.dumps(findings, ensure_ascii=False)[:800]}       # 截取800字符避免过长
        可用来源：{source_list}

        需要撰写的章节：{json.dumps(sections, ensure_ascii=False)}

        请返回一个JSON对象，键为章节名，值为对应内容。例如：
        {{
            "引言": "内容...",
            "文献综述": "内容...",
            ...
        }}
        只返回JSON。"""

        response = llm_call_with_retry([HumanMessage(content=synthesis_prompt)])
        draft_sections = safe_parse_json(response.content, {})

        # 确保每个章节都有内容，如果缺失则填入占位文本
        for section in sections:
            if section not in draft_sections or not draft_sections[section]:
                draft_sections[section] = f"关于{section}的内容待补充。"

        logger.info(f"已生成 {len(draft_sections)} 个章节")
        return {
            "draft_sections": draft_sections,
            "current_phase": "report_generation",
            "messages": [AIMessage(content="章节草稿生成完成")]
        }

    # ---- 5. 报告生成节点 ----
    def report_generation_node(state: ResearchState) -> dict:
        """将各章节整合为结构化的最终研究报告，并生成参考文献列表"""
        logger.info("报告生成阶段...")
        topic = state["research_topic"]
        outline = state.get("outline", {})
        draft = state.get("draft_sections", {})
        search_results = state.get("search_results", [])

        # 生成参考文献列表
        citations = []
        for i, result in enumerate(search_results[:6]):   # 最多6条引用
            citations.append({
                "id": f"[{i+1}]",
                "authors": result.get("authors", ["Unknown"]),
                "title": result.get("title", ""),
                "source": result.get("source", ""),
                "year": result.get("year", 2024),
                "url": result.get("url", "")
            })

        # 让 LLM 把草稿整合为标准研究报告格式（尝试结构化输出）
        report_prompt = f"""请将以下内容整合为一份规范的研究报告，并使用JSON格式返回，严格遵循提供的结构。

        标题：{outline.get('title', topic)}
        摘要：{outline.get('abstract', '')}
        章节内容：{json.dumps(draft, ensure_ascii=False)[:2000]}    # 截取2000字符避免token超限
        参考文献：{json.dumps(citations, ensure_ascii=False)[:500]}

        返回的JSON必须包含如下字段（字段名与英文完全一致）：
        - title (字符串)
        - executive_summary (字符串)
        - introduction (字符串)
        - methodology (字符串)
        - findings (字符串列表)
        - analysis (字符串)
        - conclusions (字符串列表)
        - recommendations (字符串列表)
        - citations (列表，每个元素包含 id, authors, title, source, year, url)
        - generated_at (当前时间字符串)

        只返回JSON。"""

        try:
            response = llm_call_with_retry([HumanMessage(content=report_prompt)])
            report_data = safe_parse_json(response.content, None)

            if not report_data:
                raise ValueError("JSON解析为空")        # 触发异常，使用回退方案

            # 补充缺失的默认值
            report_data.setdefault("citations", citations)
            report_data.setdefault("generated_at", datetime.now().isoformat())
            report_data.setdefault("title", outline.get("title", topic))

            # 拼接 Markdown 格式的最终报告文本
            final_text = f"# {report_data.get('title', '')}\n\n"
            final_text += f"## 摘要\n{report_data.get('executive_summary', outline.get('abstract', ''))}\n\n"
            final_text += f"## 引言\n{report_data.get('introduction', '')}\n\n"
            final_text += f"## 方法论\n{report_data.get('methodology', outline.get('methodology', ''))}\n\n"
            final_text += "## 发现\n" + "\n".join(f"- {f}" for f in report_data.get('findings', [])) + "\n\n"
            final_text += f"## 分析\n{report_data.get('analysis', '')}\n\n"
            final_text += "## 结论\n" + "\n".join(f"- {c}" for c in report_data.get('conclusions', [])) + "\n\n"
            final_text += "## 建议\n" + "\n".join(f"- {r}" for r in report_data.get('recommendations', [])) + "\n\n"
            final_text += "## 参考文献\n"
            for c in report_data.get('citations', []):
                authors = ', '.join(c.get('authors', ['Unknown']))
                final_text += f"{c.get('id', '')} {authors}. {c.get('title', '')}. {c.get('source', '')}, {c.get('year', '')}.\n"

            final_report = final_text
        except Exception as e:
            logger.error(f"结构化报告生成失败，回退到简单拼接: {e}")
            # 回退方案：直接按章节拼接
            report_sections = [f"# {outline.get('title', topic)}", f"## 摘要\n{outline.get('abstract', '')}"]
            for section_title, content in draft.items():
                report_sections.append(f"## {section_title}\n{content}")
            report_sections.append("## 参考文献")
            for c in citations:
                authors = ", ".join(c.get("authors", ["Unknown"]))
                report_sections.append(f"{c['id']} {authors}. {c['title']}. {c['source']}, {c['year']}.")
            final_report = "\n\n".join(report_sections)

        logger.info(f"报告生成完成，字数: {len(final_report)}")
        return {
            "final_report": final_report,
            "citations": citations,
            "current_phase": "quality_check",
            "messages": [AIMessage(content="研究报告已生成")]
        }

    # ---- 6. 质量检查节点 ----
    def quality_check_node(state: ResearchState) -> dict:
        """评估报告质量（0-10分），并给出反馈；如果不满意且未超过迭代次数，将返回信息收集重新搜索"""
        logger.info("质量检查阶段...")
        report = state.get("final_report", "")
        citations = state.get("citations", [])

        quality_prompt = f"""评估以下研究报告质量（0-10分）并给出改进建议。
        报告前500字：{report[:500]}
        引用数量：{len(citations)}
        评估维度：结构、逻辑、引用、语言、学术性。
        返回JSON：{{"score": 8.5, "feedback": "具体改进建议"}}"""

        response = llm_call_with_retry([HumanMessage(content=quality_prompt)])
        eval_data = safe_parse_json(response.content, {"score": 7.5, "feedback": "整体尚可"})
        quality_score = float(eval_data.get("score", 7.5))   # 转换为浮点数
        feedback = eval_data.get("feedback", "")

        logger.info(f"质量评分: {quality_score}/10")
        return {
            "quality_score": quality_score,
            "quality_feedback": feedback,
            "current_phase": "completed",
            "iteration_count": state.get("iteration_count", 0) + 1,   # 迭代次数+1
            "messages": [AIMessage(content=f"质量评估：{quality_score}/10。反馈：{feedback}")]
        }

    # ---- 7. 路由函数：决定是否迭代 ----
    def should_continue(state: ResearchState) -> Literal["continue", "complete"]:
        """如果质量评分 >= 7.5 或已迭代 3 次，则结束；否则返回信息收集重新搜索更高质量资料"""
        score = state.get("quality_score", 0)
        iterations = state.get("iteration_count", 0)
        if score >= 7.5 or iterations >= 3:
            return "complete"                  # 质量合格或已达最大迭代次数
        return "continue"                      # 需要继续改进

    # ==================== 构建状态图 ====================
    graph = StateGraph(ResearchState)

    # 添加节点
    graph.add_node("planning", planning_node)
    graph.add_node("information_gathering", information_gathering_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("quality_check", quality_check_node)

    # 定义边：顺序执行
    graph.add_edge(START, "planning")                # 开始 -> 规划
    graph.add_edge("planning", "information_gathering")  # 规划 -> 信息收集
    graph.add_edge("information_gathering", "analysis")  # 信息收集 -> 分析
    graph.add_edge("analysis", "synthesis")              # 分析 -> 综合
    graph.add_edge("synthesis", "report_generation")     # 综合 -> 报告生成
    graph.add_edge("report_generation", "quality_check") # 报告生成 -> 质量检查

    # 条件边：质量检查后根据 should_continue 决定继续迭代还是结束
    graph.add_conditional_edges(
        "quality_check",
        should_continue,
        {
            "continue": "information_gathering",   # 重新收集资料（可能搜索到新资料）
            "complete": END
        }
    )

    # 使用内存检查点保存状态，方便跟踪和恢复（未在演示中完全利用）
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# %%
# ==================== 第十部分：运行研究任务 ====================
def run_research(topic: str):
    """启动研究助手，针对给定主题执行整个研究流程，并打印报告和统计信息"""
    logger.info("=" * 60)
    logger.info(f"启动研究任务: {topic}")

    try:
        assistant = create_research_assistant()
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
            "research_topic": topic,
            "research_questions": [],
            "search_results": [],
            "analyzed_sources": [],
            "outline": {},
            "findings": [],
            "draft_sections": {},
            "final_report": "",
            "citations": [],
            "current_phase": "planning",
            "iteration_count": 0,
            "quality_score": 0.0,
            "quality_feedback": ""
        }

        # 配置唯一 thread_id 以支持检查点
        config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
        result = assistant.invoke(initial_state, config)

        # 打印报告
        print("\n" + "=" * 60)
        print("研究报告")
        print("=" * 60)
        print(result.get("final_report", "报告生成失败"))

        # 打印参考文献
        print("\n" + "-" * 60)
        print("参考文献")
        print("-" * 60)
        for citation in result.get("citations", []):
            authors = ", ".join(citation.get("authors", ["Unknown"]))
            print(f"{citation['id']} {authors}. {citation['title']}. {citation['source']}, {citation['year']}.")

        # 打印研究统计信息
        print("\n" + "-" * 60)
        print("研究统计")
        print("-" * 60)
        print(f"  - 收集资料数: {len(result.get('search_results', []))}")
        print(f"  - 分析来源数: {len(result.get('analyzed_sources', []))}")
        print(f"  - 迭代次数: {result.get('iteration_count', 0)}")
        print(f"  - 质量评分: {result.get('quality_score', 0):.1f}/10")
        print(f"  - 报告字数: {len(result.get('final_report', ''))}")

        return result
    except Exception as e:
        logger.error(f"研究任务失败: {e}", exc_info=True)
        print(f"\n研究任务执行失败：{e}")
        return None

# %%
# ==================== 主程序入口 ====================
def main():
    logger.info("智能研究助手系统启动")
    research_topics = ["人工智能在医疗诊断中的应用"]     # 可以修改或增加研究主题
    for topic in research_topics:
        run_research(topic)                             # 对每个主题执行研究流程
    logger.info("所有任务完成")

if __name__ == "__main__":
    main()                                              # 如果直接运行本脚本，则调用 main 函数