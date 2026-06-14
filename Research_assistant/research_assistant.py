# Jupyter 单元格分隔符（无实际影响，仅用于在 Jupyter Notebook 中划分单元格）
# %%

# ==================== 第一部分：导入库 ====================
# 导入操作系统接口模块，用于读取环境变量、处理文件路径等
import os
# 导入 sys 模块，用于修改 Python 解释器运行时的环境（如添加模块搜索路径）
import sys
# 导入 JSON 模块，用于序列化和反序列化 JSON 数据
import json
# 导入日志模块，用于记录程序运行时的调试、信息、警告和错误信息
import logging
# 从 typing 模块导入类型注解工具，用于定义复杂数据类型，提高代码可读性和类型安全性
from typing import TypedDict, Literal, Annotated, Optional
# 导入 datetime 模块，用于获取当前时间，生成报告时间戳
from datetime import datetime

# 将当前文件的父目录（项目根目录）添加到 Python 的模块搜索路径中，以便能够导入 shared 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从 shared 包导入以下工具函数：
# safe_parse_json: 安全解析 JSON 字符串
# setup_logging: 配置日志记录器
# ModelCache: 线程安全的 LLM 模型缓存类
# llm_invoke_with_retry: 带重试机制的 LLM 调用函数
from shared import safe_parse_json, setup_logging, ModelCache, llm_invoke_with_retry

# 导入 python-dotenv 库，用于从 .env 文件加载环境变量（如 API 密钥）
import dotenv
# 导入 LangChain 中的人类消息和 AI 消息类，用于构建对话提示
from langchain_core.messages import HumanMessage, AIMessage
# 导入 LangGraph 状态图构建所需的核心类：StateGraph、起始节点 START、结束节点 END、消息合并函数 add_messages
from langgraph.graph import StateGraph, START, END, add_messages
# 导入内存检查点保存器，用于保存图执行过程中的状态，支持回溯和恢复
from langgraph.checkpoint.memory import MemorySaver

# %% 
# ==================== 第二部分：日志配置 ====================
# 初始化当前模块的日志记录器，使用模块名作为日志名称
logger = setup_logging(__name__)

# %% 
# ==================== 第三部分：JSON 解析辅助函数 ====================
# safe_parse_json 已从 shared.utils 导入，此处不再重复定义，仅作说明

# %% 
# ==================== 第四部分：环境变量与模型单例 ====================
# 加载 .env 文件中的环境变量（例如 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL）
dotenv.load_dotenv()

# 从环境变量中获取 DeepSeek API 密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 从环境变量中获取 DeepSeek API 的基础 URL
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建全局的模型缓存实例，设定生成温度为 0.3，最大输出 token 数为 2000
_model_cache = ModelCache(temperature=0.3, max_tokens=2000)

# 验证 DeepSeek API 密钥是否已设置，若未设置则抛出错误，阻止程序继续运行
if not DEEPSEEK_API_KEY:
    raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置，请在 .env 文件中添加该变量。")
# 验证 DeepSeek API 基础 URL 是否已设置，若未设置则抛出错误
if not DEEPSEEK_BASE_URL:
    raise ValueError("环境变量 DEEPSEEK_BASE_URL 未设置，请在 .env 文件中添加该变量。")

def get_model():
    """获取共享的 LLM 模型实例（通过模型缓存获取）"""
    return _model_cache.get()

# %% 
# ==================== 第五部分：研究状态定义 ====================
# 使用 TypedDict 定义一个研究助手工作流中共享的状态字典结构
class ResearchState(TypedDict):
    """研究助手工作流中共享的状态，贯穿所有节点"""
    # 消息历史列表，使用 add_messages 函数来合并新的消息（追加而非覆盖）
    messages: Annotated[list, add_messages]
    # 当前的研究主题，由用户输入
    research_topic: str
    # 研究过程中需要回答的关键问题列表
    research_questions: list[str]
    # 原始搜索结果列表，存储从数据源检索到的文献或新闻条目
    search_results: list[dict]
    # 经过分析整理的来源摘要列表
    analyzed_sources: list[dict]
    # 研究大纲，包含标题、摘要、章节等
    outline: dict
    # 关键发现列表，每个发现包含主题、证据、置信度等
    findings: list[dict]
    # 各章节的草稿内容，字典的键为章节名，值为文本内容
    draft_sections: dict
    # 最终生成的研究报告全文（Markdown 格式）
    final_report: str
    # 参考文献列表
    citations: list[dict]
    # 当前所处的阶段名称（如 planning, information_gathering 等）
    current_phase: str
    # 迭代计数器，记录质量检查后重新收集资料的次数
    iteration_count: int
    # 报告质量评分，范围 0-10
    quality_score: float
    # 质量反馈文本，用于指导下一次迭代改进
    quality_feedback: str

# %% 
# ==================== 第六部分：模拟数据源 ====================
# 模拟学术数据库，键为主题领域，值为该领域的论文列表
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

# 模拟网络搜索结果，结构类似学术数据库
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
# ==================== 第七部分：工具函数 ====================
def search_academic_database(topic: str, max_results: int = 5) -> list[dict]:
    """
    从模拟学术数据库中搜索与主题相关的论文。
    使用双向模糊匹配：主题词与数据库键互包含即认为匹配。
    """
    results = []                                    # 初始化空列表，用于存储匹配的论文
    for key, papers in ACADEMIC_DATABASE.items():  # 遍历学术数据库中的每个主题和对应的论文列表
        # 双向模糊匹配：用户输入的主题包含数据库键，或数据库键包含用户输入的主题（不区分大小写）
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            for paper in papers[:max_results]:      # 只取前 max_results 篇论文
                # 将论文信息复制到新字典，并添加类型和固定的相关性评分
                results.append({**paper, "type": "academic", "relevance_score": 0.9})
    return results[:max_results]                    # 返回最多 max_results 条结果

def search_web(topic: str, max_results: int = 5) -> list[dict]:
    """从模拟网络新闻中搜索与主题相关的文章，逻辑与学术搜索类似。"""
    results = []
    for key, items in WEB_SEARCH_RESULTS.items():
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            for item in items[:max_results]:
                results.append({**item, "type": "web", "relevance_score": 0.8})
    return results[:max_results]

# ==================== LLM 调用重试机制 ====================
def llm_call_with_retry(prompt_messages, max_retries=3, delay=1.5):
    """
    带有重试机制的 LLM 调用包装函数。
    直接委托给共享模块中的 llm_invoke_with_retry，传入模型缓存实例。
    """
    return llm_invoke_with_retry(_model_cache, prompt_messages, max_retries, delay)

# %% 
# ==================== 第八部分：智能体节点与图构建 ====================
def create_research_assistant():
    """
    创建并返回编译好的研究助手状态图。
    工作流程：
    规划 → 信息收集 → 分析 → 综合 → 报告生成 → 质量检查
    如果质量不合格且未超过最大迭代次数，则返回信息收集节点重新搜索更高质量资料。
    """

    # ---- 1. 研究规划节点 ----
    def planning_node(state: ResearchState) -> dict:
        """根据研究主题，让 LLM 生成研究计划（标题、大纲、关键问题等）"""
        logger.info("研究规划阶段...")                     # 记录阶段开始日志
        topic = state["research_topic"]                    # 从状态中获取研究主题

        # 构建研究规划的提示词，要求 LLM 返回严格的 JSON 格式
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

        # 调用 LLM 并传入人类消息
        response = llm_call_with_retry([HumanMessage(content=planning_prompt)])
        # 安全解析 LLM 返回的 JSON，如果解析失败则使用默认大纲
        outline = safe_parse_json(response.content, {
            "title": f"{topic}研究",
            "abstract": f"本研究探讨{topic}的相关问题。",
            "sections": ["引言", "文献综述", "研究方法", "结果分析", "结论"],
            "key_questions": [f"{topic}的现状如何？", f"{topic}的发展趋势是什么？", f"{topic}面临哪些挑战？"],
            "methodology": "文献研究与案例分析相结合"
        })

        # 记录规划结果
        logger.info(f"标题: {outline.get('title')}, 章节数: {len(outline.get('sections', []))}")
        # 返回更新后的状态字典
        return {
            "outline": outline,
            "research_questions": outline.get("key_questions", []),   # 从大纲中提取研究问题列表
            "current_phase": "information_gathering",                 # 设置下一阶段为信息收集
            "messages": [AIMessage(content=f"研究计划已制定：{outline.get('title')}")]  # 添加一条 AI 消息记录
        }

    # ---- 2. 信息收集节点 ----
    def information_gathering_node(state: ResearchState) -> dict:
        """模拟信息检索：同时调用学术数据库和网络搜索，收集相关资料"""
        logger.info("信息收集阶段...")
        topic = state["research_topic"]                    # 获取研究主题
        academic = search_academic_database(topic)         # 搜索学术文献
        web = search_web(topic)                            # 搜索网络资源
        all_results = academic + web                       # 合并两个来源的结果
        logger.info(f"收集到 {len(all_results)} 条资料")
        if not all_results:                                # 如果没有找到任何资料，记录警告
            logger.warning("未找到相关资料，报告可能缺乏依据")
        return {
            "search_results": all_results,                 # 将合并后的搜索结果存入状态
            "current_phase": "analysis",                   # 下一阶段设置为分析
            "messages": [AIMessage(content=f"已收集 {len(all_results)} 条相关资料")]
        }

    # ---- 3. 信息分析节点 ----
    def analysis_node(state: ResearchState) -> dict:
        """对收集的资料进行深度分析，提取关键发现、证据和信息缺口"""
        logger.info("信息分析阶段...")
        topic = state["research_topic"]                    # 研究主题
        search_results = state.get("search_results", [])   # 获取原始搜索结果，默认为空列表
        questions = state.get("research_questions", [])    # 获取研究问题列表
        quality_feedback = state.get("quality_feedback", "") # 获取上一次迭代的质量反馈（可能为空）

        # 如果有质量反馈（来自质量检查节点的迭代建议），将其作为分析的提示
        feedback_section = ""
        if quality_feedback:
            feedback_section = f"\n\n本次为迭代改进，请特别针对以下反馈调整分析：\n{quality_feedback}"

        # 构造资料摘要，最多取前 8 条，避免提示词过长
        sources_summary = "\n".join([f"- {r['title']}: {r.get('snippet', '')}" for r in search_results[:8]])

        # 构建分析提示词
        analysis_prompt = f"""基于以下资料，对研究主题进行深入分析：
        研究主题：{topic}
        核心问题：\n""" + "\n".join(f"- {q}" for q in questions) + f"""
        资料：{sources_summary}{feedback_section}

        请提供 JSON 格式输出，包含：
        - key_findings: 列表，每个元素包含 finding, evidence, confidence, sources
        - analysis_points: 列表，观点比较等
        - information_gaps: 列表
        只返回 JSON。"""

        # 调用 LLM 进行分析
        response = llm_call_with_retry([HumanMessage(content=analysis_prompt)])
        # 安全解析 JSON，如果失败则得到空字典
        analysis_data = safe_parse_json(response.content, {})

        # 提取 LLM 返回的 key_findings，如果为空则构造一个默认发现
        key_findings = analysis_data.get("key_findings", [])
        if not key_findings:
            key_findings = [{
                "finding": f"关于{topic}的初步发现",
                "evidence": "综合资料显示",
                "confidence": 0.8,
                "sources": [r["title"] for r in search_results[:3]]
            }]

        # 将 key_findings 整理成统一的结构
        findings = []
        for f in key_findings:
            findings.append({
                "topic": topic,
                "key_points": [f.get("finding", "")],
                # 确保证据是列表格式
                "evidence": [f.get("evidence", "")] if isinstance(f.get("evidence"), str) else f.get("evidence", []),
                "confidence": f.get("confidence", 0.7),
                "sources": f.get("sources", [])
            })

        # 构建分析后的来源摘要列表
        analyzed_sources = []
        for i, result in enumerate(search_results[:6]):   # 最多处理 6 个来源
            analyzed_sources.append({
                "id": f"src_{i+1}",
                "title": result["title"],
                "key_takeaways": result.get("snippet", "")[:100],  # 截取前100个字符作为关键点
                "relevance": result.get("relevance_score", 0.5)
            })

        logger.info(f"提取了 {len(findings)} 组关键发现")
        return {
            "findings": findings,                               # 存储关键发现
            "analyzed_sources": analyzed_sources,              # 存储分析后的来源
            "current_phase": "synthesis",                      # 下一阶段：综合
            "messages": [AIMessage(content="分析完成")]
        }

    # ---- 4. 知识综合节点（生成各章节草稿） ----
    def synthesis_node(state: ResearchState) -> dict:
        """根据分析结果，为报告的每个章节撰写草稿内容"""
        logger.info("知识综合阶段...")
        topic = state["research_topic"]
        outline = state.get("outline", {})                     # 获取大纲
        findings = state.get("findings", [])                    # 获取关键发现
        sources = state.get("analyzed_sources", [])             # 获取分析后的来源

        # 从大纲中获取章节列表，若未提供则使用默认章节
        sections = outline.get("sections", ["引言", "方法", "发现", "结论"])
        # 构造来源列表摘要，最多 5 个
        source_list = "\n".join([f"- {s['title']}" for s in sources[:5]])

        # 构建综合提示词，要求 LLM 为每个章节生成 150-300 字内容
        synthesis_prompt = f"""你是专业报告撰写人。请根据以下信息，为研究报告的每个章节撰写内容（每章150-300字）。

        研究主题：{topic}
        大纲摘要：{outline.get('abstract', '')}
        关键发现：{json.dumps(findings, ensure_ascii=False)[:800]}       # 截取800字符，防止 token 过多
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
        draft_sections = safe_parse_json(response.content, {})   # 安全解析 JSON 获得章节草稿

        # 确保每一个章节都有内容，如果缺失则写入占位文本
        for section in sections:
            if section not in draft_sections or not draft_sections[section]:
                draft_sections[section] = f"关于{section}的内容待补充。"

        logger.info(f"已生成 {len(draft_sections)} 个章节")
        return {
            "draft_sections": draft_sections,                # 存储各章节草稿
            "current_phase": "report_generation",            # 下一阶段：报告生成
            "messages": [AIMessage(content="章节草稿生成完成")]
        }

    # ---- 5. 报告生成节点 ----
    def report_generation_node(state: ResearchState) -> dict:
        """将各章节整合为结构化的最终研究报告，并生成参考文献列表"""
        logger.info("报告生成阶段...")
        topic = state["research_topic"]
        outline = state.get("outline", {})
        draft = state.get("draft_sections", {})              # 获取各章节草稿
        search_results = state.get("search_results", [])     # 获取原始搜索结果，用于生成引用

        # 根据搜索结果生成参考文献列表，最多 6 条
        citations = []
        for i, result in enumerate(search_results[:6]):
            citations.append({
                "id": f"[{i+1}]",                            # 引用编号，如 [1]
                "authors": result.get("authors", ["Unknown"]),
                "title": result.get("title", ""),
                "source": result.get("source", ""),
                "year": result.get("year", 2024),
                "url": result.get("url", "")
            })

        # 构建报告生成提示词，要求 LLM 输出符合 ResearchReport 结构的 JSON
        report_prompt = f"""请将以下内容整合为一份规范的研究报告，并使用JSON格式返回，严格遵循提供的结构。

        标题：{outline.get('title', topic)}
        摘要：{outline.get('abstract', '')}
        章节内容：{json.dumps(draft, ensure_ascii=False)[:2000]}    # 截取2000字符
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
            report_data = safe_parse_json(response.content, None)  # 尝试解析 LLM 返回的 JSON

            if not report_data:
                raise ValueError("JSON解析为空")          # 如果解析结果为空，抛出异常进入回退方案

            # 为可能缺失的字段提供默认值
            report_data.setdefault("citations", citations)
            report_data.setdefault("generated_at", datetime.now().isoformat())
            report_data.setdefault("title", outline.get("title", topic))

            # 将结构化数据拼接成 Markdown 格式的最终报告
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
            # 如果 LLM 输出的 JSON 无法解析或其它异常，回退到简单的章节拼接
            logger.error(f"结构化报告生成失败，回退到简单拼接: {e}")
            report_sections = [f"# {outline.get('title', topic)}", f"## 摘要\n{outline.get('abstract', '')}"]
            for section_title, content in draft.items():
                report_sections.append(f"## {section_title}\n{content}")
            report_sections.append("## 参考文献")
            for c in citations:
                authors = ", ".join(c.get("authors", ["Unknown"]))
                report_sections.append(f"{c['id']} {authors}. {c['title']}. {c['source']}, {c['year']}.")
            final_report = "\n\n".join(report_sections)      # 用两个换行连接各节

        logger.info(f"报告生成完成，字数: {len(final_report)}")
        return {
            "final_report": final_report,                    # 存储最终报告
            "citations": citations,                          # 存储引用列表
            "current_phase": "quality_check",               # 下一阶段：质量检查
            "messages": [AIMessage(content="研究报告已生成")]
        }

    # ---- 6. 质量检查节点 ----
    def quality_check_node(state: ResearchState) -> dict:
        """评估报告质量（0-10分），并给出反馈；如果质量不达标且未超过最大迭代次数，将触发信息收集重新搜索"""
        logger.info("质量检查阶段...")
        report = state.get("final_report", "")               # 获取报告全文
        citations = state.get("citations", [])               # 获取引用列表

        # 构建质量评估提示词，只发送报告前 500 字符供评估
        quality_prompt = f"""评估以下研究报告质量（0-10分）并给出改进建议。
        报告前500字：{report[:500]}
        引用数量：{len(citations)}
        评估维度：结构、逻辑、引用、语言、学术性。
        返回JSON：{{"score": 8.5, "feedback": "具体改进建议"}}"""

        response = llm_call_with_retry([HumanMessage(content=quality_prompt)])
        # 解析 LLM 返回的评分与反馈，默认分数 7.5
        eval_data = safe_parse_json(response.content, {"score": 7.5, "feedback": "整体尚可"})
        quality_score = float(eval_data.get("score", 7.5))   # 将评分转换为浮点数
        feedback = eval_data.get("feedback", "")

        logger.info(f"质量评分: {quality_score}/10")
        return {
            "quality_score": quality_score,                         # 记录评分
            "quality_feedback": feedback,                           # 记录反馈
            "current_phase": "completed",                           # 标记当前节点执行完毕
            "iteration_count": state.get("iteration_count", 0) + 1, # 迭代次数加 1
            "messages": [AIMessage(content=f"质量评估：{quality_score}/10。反馈：{feedback}")]
        }

    # ---- 7. 路由函数：决定是否继续迭代 ----
    def should_continue(state: ResearchState) -> Literal["continue", "complete"]:
        """如果质量评分 >= 7.5 或已迭代 3 次，则结束工作流；否则返回信息收集节点重新搜索资料"""
        score = state.get("quality_score", 0)
        iterations = state.get("iteration_count", 0)
        if score >= 7.5 or iterations >= 3:
            return "complete"           # 质量合格或已达最大迭代次数，流程结束
        return "continue"               # 需要继续改进，返回信息收集

    # ==================== 构建状态图 ====================
    # 创建一个以 ResearchState 为状态类型的状态图
    graph = StateGraph(ResearchState)

    # 向图中添加各功能节点
    graph.add_node("planning", planning_node)
    graph.add_node("information_gathering", information_gathering_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("quality_check", quality_check_node)

    # 定义节点之间的有向边，形成顺序执行流程
    graph.add_edge(START, "planning")                              # 从开始节点到规划节点
    graph.add_edge("planning", "information_gathering")            # 规划完成 → 信息收集
    graph.add_edge("information_gathering", "analysis")           # 信息收集完成 → 分析
    graph.add_edge("analysis", "synthesis")                       # 分析完成 → 综合
    graph.add_edge("synthesis", "report_generation")              # 综合完成 → 报告生成
    graph.add_edge("report_generation", "quality_check")          # 报告生成完成 → 质量检查

    # 添加条件边：从质量检查节点出发，根据 should_continue 函数的返回值决定下一跳
    graph.add_conditional_edges(
        "quality_check",
        should_continue,
        {
            "continue": "information_gathering",  # 返回信息收集节点，形成迭代循环
            "complete": END                       # 流程结束
        }
    )

    # 创建内存检查点保存器，用于记录图执行的状态历史（演示中未完全利用）
    memory = MemorySaver()
    # 编译状态图并返回一个可执行的应用实例，附带检查点功能
    return graph.compile(checkpointer=memory)

# %% 
# ==================== 第九部分：运行研究任务 ====================
def run_research(topic: str):
    """启动研究助手，针对给定主题执行整个研究流程，并打印报告和统计信息"""
    logger.info("=" * 60)
    logger.info(f"启动研究任务: {topic}")

    try:
        assistant = create_research_assistant()     # 创建并获取编译好的研究助手图应用
        # 准备图的初始状态
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

        # 配置运行的 thread_id，用于检查点追踪（每个任务生成唯一的 ID）
        config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
        # 调用图应用执行工作流，传入初始状态和配置，获取最终状态
        result = assistant.invoke(initial_state, config)

        # 打印最终报告
        print("\n" + "=" * 60)
        print("研究报告")
        print("=" * 60)
        print(result.get("final_report", "报告生成失败"))    # 获取报告文本，若没有则显示失败信息

        # 打印参考文献
        print("\n" + "-" * 60)
        print("参考文献")
        print("-" * 60)
        for citation in result.get("citations", []):         # 遍历引用列表并格式化输出
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

        return result                                 # 返回最终状态字典，供外部使用
    except Exception as e:
        logger.error(f"研究任务失败: {e}", exc_info=True)  # 记录异常详细信息
        print(f"\n研究任务执行失败：{e}")
        return None                                  # 异常时返回 None

# %% 
# ==================== 主程序入口 ====================
def main():
    """命令行演示入口：启动研究助手，对预定义的主题列表执行研究流程"""
    logger.info("智能研究助手系统启动")
    research_topics = ["人工智能在医疗诊断中的应用"]    # 预定义的研究主题列表，可根据需要扩充
    for topic in research_topics:                       # 遍历每个主题
        run_research(topic)                             # 调用 run_research 执行研究
    logger.info("所有任务完成")

# 判断当前模块是否作为主程序运行，而不是被导入
if __name__ == "__main__":
    main()                                              # 如果是主程序，则执行 main 函数