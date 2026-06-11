
# AI Applications Hub

基于 **LangChain + LangGraph** 构建的三大高级 AI 应用，覆盖**智能客服**、**知识问答**、**学术研究**核心场景。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-orange.svg)](https://www.deepseek.com/)

---

## 📖 目录

- [项目概览](#-项目概览)
- [一、多代理智能客服系统](#一多代理智能客服系统)
- [二、RAG 智能问答系统](#二rag-智能问答系统)
- [三、智能研究助手](#三智能研究助手)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [项目仓库](#-项目仓库)

---

## 📦 项目概览

| 系统 | 定位 | 核心技术 | 核心亮点 |
|------|------|----------|----------|
| **多代理智能客服系统** | 企业级智能客服 | LangGraph 多 Agent 协作 | 意图路由、工具分配、质量检查、人工升级 |
| **RAG 智能问答系统** | 知识库问答引擎 | LangChain + Chroma | 双重去重、查询改写、低分重试闭环 |
| **智能研究助手** | 自动化研究平台 | LangGraph 多阶段工作流 | 六阶段流程、自反馈迭代、多源检索 |

---

## 一、多代理智能客服系统

**定位**：企业级智能客服解决方案

### 核心架构

- **意图分类器**：LLM 识别用户需求（技术支持 / 订单服务 / 产品咨询 / 通用对话 / 升级人工），输出置信度
- **四大专业 Agent**：
  - 技术支持 Agent：故障排除、FAQ 搜索
  - 订单服务 Agent：订单查询、物流跟踪
  - 产品咨询 Agent：产品推荐、价格查询
  - 通用对话 Agent：闲聊兜底、二次确认
- **质量检查器**：四维度百分制评估，低分自动追加人工转接提示
- **双通道升级**：用户主动要求直接触发；系统质检低分自动触发

### 工作流程

```
用户消息 → 意图分类 → 路由决策 → Agent 处理 → 质量检查 → 升级判断 → 最终响应
```

**路由策略**：置信度 < 0.6 降级至通用对话兜底，避免错误路由；`escalate` 意图直接跳过质检转人工。

### 适用场景

在线客服、技术支持、订单查询、产品咨询

---

## 二、RAG 智能问答系统

**定位**：基于知识库的智能问答引擎

### 核心功能

- **文档处理管线**：加载 → 递归分块（500/100）→ MD5 精确去重 → 向量语义去重 → Chroma 索引
- **查询改写**：结合对话历史消除指代歧义，将模糊问题转为独立检索查询
- **智能检索**：余弦相似度搜索，Top‑K 文档召回
- **来源追溯**：返回引用文档、内容预览、元数据
- **置信度评估**：独立 Evaluator 节点打分，低于 0.6 分触发二次检索重试

### 处理流程

```
用户提问 → 查询重写 → 文档检索 → 生成回答 → 置信度评估 → 返回结果
                                                    │
                                              低于0.6分？
                                                    │
                                              是 → 二次检索重试
                                              否 → 返回最终答案
```

### 适用场景

企业知识库、文档检索、教育辅助、内部问答

---

## 三、智能研究助手

**定位**：自动化研究平台

### 六阶段流程

| 阶段 | 操作 | 产出 |
|------|------|------|
| 1. 规划 | LLM 生成研究大纲、章节、关键问题 | 研究大纲 JSON |
| 2. 收集 | 学术数据库 + 网络新闻双源并发检索 | 原始资料列表 |
| 3. 分析 | 提炼关键发现、证据、信息空白 | 分析结果 + 来源摘要 |
| 4. 综合 | 为每个章节撰写草稿，标注引用 | 章节草稿字典 |
| 5. 生成 | 整合为完整 Markdown 报告 + 参考文献 | 最终报告文本 |
| 6. 检查 | 10 分制质量评估 | 评分 + 改进建议 |

### 迭代机制

评分 < 7.5 分时，携带质量反馈回到阶段 2 重新检索补充资料，再走一遍分析 → 综合 → 生成 → 检查流程。最多迭代 2 次，超过后强制输出。

### 适用场景

学术研究、市场调研、技术趋势分析、报告撰写

---

## 🧱 技术架构

### 共同技术栈

| 层级 | 技术选型 |
|------|----------|
| 核心框架 | LangChain + LangGraph |
| 语言模型 | DeepSeek（ChatOpenAI 兼容接口） |
| 数据验证 | Pydantic |
| 向量存储 | Chroma |
| 后端服务 | FastAPI + Uvicorn |
| 嵌入模型 | 智谱 embedding-2 / sentence-transformers（后备） |

### 系统设计原则

- **智能路由**：条件边实现动态分支与循环
- **质量兜底**：每个系统内置评估节点，低分触发重试或升级
- **状态管理**：LangGraph State 集中管理，节点输入输出可追溯
- **工具可扩展**：标准化工具接口，模拟数据可随时替换为真实 API
- **防御性设计**：安全 JSON 解析、LLM 调用重试、全局异常捕获

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（推荐）或智谱 API Key

### 安装

```bash
git clone https://github.com/lingdugongsheng/AI.git
cd AI
pip install -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 运行

```bash
# 启动客服系统 API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 或运行各子系统 Demo
python multi_agent.py
python rag.py
python reserch_assistant.py
```

---

## 📁 项目仓库

```
AI/
├── multi_agent.py           # 多代理智能客服系统核心
├── rag.py                   # RAG 智能问答系统核心
├── reserch_assistant.py     # 智能研究助手核心
├── main.py                  # FastAPI 服务入口
├── .env                     # 环境变量配置
├── requirements.txt         # 依赖列表
└── README.md                # 本文件
```

每个子系统配有独立的 README 文件，包含详细的架构图、API 文档和设计说明。

---

## 🎯 应用价值

- **智能客服**：多 Agent 协作处理复杂业务，质量兜底保障用户体验，降低人工成本
- **RAG 问答**：检索增强减少模型幻觉，来源可追溯，知识库可随时更新
- **研究助手**：自动化研究全流程，自反馈迭代提升报告质量，节省研究时间

---

本项目为学习与演示用途，所有模块均提供可运行的示例代码与详细配置说明。模型 API 调用需遵循对应服务商的使用协议。
```
