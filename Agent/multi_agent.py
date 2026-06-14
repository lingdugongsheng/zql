# %%
# ==================== 第一部分：导入必要的库 ====================
import os, sys, json, logging
from typing import List, Dict, Any, TypedDict, Literal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import safe_parse_json, setup_logging, ModelCache, llm_invoke_with_retry

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

# %%
# ==================== 第二部分：配置日志与环境变量 ====================
logger = setup_logging(__name__)
_model_cache = ModelCache(temperature=0.2, max_tokens=1000)

def get_model():
    """Get the shared model instance."""
    return _model_cache.get()


# %%
# ==================== 第三部分：模拟数据 ====================
MOCK_ORDERS = {
    "ORD001": {"status": "已发货", "product": "智能手表 Pro", "price": 1299, "shipping": "顺丰快递", "tracking": "SF1234567890", "estimated_delivery": "2024-12-20"},
    "ORD002": {"status": "处理中", "product": "无线耳机 Max", "price": 899, "shipping": "待发货", "tracking": None, "estimated_delivery": "2024-12-22"},
    "ORD003": {"status": "已完成", "product": "便携充电宝", "price": 199, "shipping": "已签收", "tracking": "YT9876543210", "estimated_delivery": "2024-12-15"}
}

MOCK_PRODUCTS = {
    "智能手表 Pro": {"price": 1299, "features": ["心率监测", "GPS定位", "防水50米", "7天续航"], "stock": 50, "rating": 4.8},
    "无线耳机 Max": {"price": 899, "features": ["主动降噪", "40小时续航", "蓝牙5.3", "通话降噪"], "stock": 120, "rating": 4.6},
    "便携充电宝": {"price": 199, "features": ["20000mAh", "快充支持", "双USB输出", "LED显示"], "stock": 200, "rating": 4.5},
    "智能音箱": {"price": 499, "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi音质"], "stock": 80, "rating": 4.7}
}

FAQ_DATABASE = {
    "连接问题": "请尝试以下步骤：1) 重启设备 2) 检查蓝牙是否开启 3) 删除配对记录后重新配对 4) 确保设备电量充足",
    "充电问题": "建议使用原装充电器，检查充电线是否损坏。如果问题持续，可能需要更换电池或送修。",
    "软件更新": "打开设备对应的APP，进入设置-关于-检查更新，按提示操作即可完成更新。",
    "退货政策": "我们支持7天无理由退货，30天内有质量问题可换货。请保留好购买凭证和完整包装。"
}


# %%
# ==================== 第四部分：工具函数 ====================
@tool
def query_order(order_id: str) -> str:
    """查询订单信息，根据订单号返回订单详情JSON字符串"""
    order = MOCK_ORDERS.get(order_id.upper())
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)
    return f"未找到订单{order_id}"


@tool
def track_shipping(tracking_number: str) -> str:
    """查询物流信息，根据快递单号返回物流状态描述"""
    if tracking_number.startswith("SF"):
        return f"顺丰快递{tracking_number}:包裹已到达配送站，预计今日送达"
    elif tracking_number.startswith("YT"):
        return f"圆通快递{tracking_number}:已签收"
    return f"未找到物流信息{tracking_number}"


@tool
def search_product(keyword: str) -> str:
    """搜索产品信息，根据关键词在产品名称中匹配，返回匹配产品列表JSON"""
    results = []
    for name, info in MOCK_PRODUCTS.items():
        if keyword.lower() in name.lower():
            results.append({"name": name, "price": info["price"], "features": info["features"], "rating": info["rating"]})
    if results:
        return json.dumps(results, ensure_ascii=False, indent=2)
    return f"未找到包含{keyword}的产品"


@tool
def get_product_recommendations(budget: int) -> str:
    """根据预算推荐产品，返回价格不超过预算且评分最高的前3款产品JSON"""
    recommendations = []
    for name, info in MOCK_PRODUCTS.items():
        if info['price'] <= budget:
            recommendations.append({"name": name, "price": info['price'], "rating": info['rating']})
    recommendations.sort(key=lambda x: x["price"], reverse=True)
    if recommendations:
        return json.dumps(recommendations[:3], ensure_ascii=False, indent=2)
    return f"在预算{budget}内暂无推荐产品"


@tool
def search_faq(problem_type: str) -> str:
    """搜索常见问题解答，根据问题类型关键词匹配FAQ答案"""
    for key, answer in FAQ_DATABASE.items():
        if problem_type in key:
            return f"【{key}】\n{answer}"
    return "未找到相关FAQ，建议联系人工客服获取更多帮助。"


# %%
# ==================== 第五部分：客服系统状态定义 ====================
class CustomerServiceState(TypedDict):
    user_message: str
    chat_history: List[Dict[str, str]]
    intent: str
    confidence: float
    agent_response: str
    needs_escalation: bool
    escalation_reason: str
    quality_score: float
    already_escalated: bool
    metadata: Dict[str, Any]


# %%
# ==================== 第六部分：安全JSON解析工具 ====================
# safe_parse_json imported from shared.utils


# %%
# ==================== 第七部分：意图分类器 ====================
class IntentClassifier:
    VALID_INTENTS = {"tech_support", "order_service", "product_consult", "general_chat", "escalate"}

    def __init__(self):
        self.llm = get_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图分类专家，分析用户消息并返回意图分类。
            
可选意图：
- tech_support: 具体技术问题、故障排除、使用帮助（如“蓝牙连不上”、“充电慢”）
- order_service: 具体订单查询、物流跟踪、退换货（如“查订单 ORD001”、“快递到哪了”）
- product_consult: 具体产品咨询、价格询问、功能介绍（如“智能手表多少钱”、“推荐一款耳机”）
- general_chat: 通用对话、闲聊、功能询问、模糊问题、非业务问题（如“你好”、“你能做什么”、“帮我写首诗”、“我不懂”）
- escalate: 明确要求人工客服、投诉、严重不满、要求经理、连续无法解决问题（如“我要投诉”、“转人工”、“叫你们经理来”）

返回格式(JSON):
{{"intent": "意图类型","confidence": 0.0-1.0, "reason": "分类原因"}}

只返回JSON，不要其他内容。"""),
            ("human", "{message}")
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm | StrOutputParser()
        result = llm_invoke_with_retry(chain, {"message": message})
        default_result = {"intent": "general_chat", "confidence": 0.5, "reason": "解析失败"}
        parsed = safe_parse_json(result, default_result)
        intent = parsed.get("intent", "general_chat")
        if intent not in self.VALID_INTENTS:
            intent = "general_chat"
        parsed["intent"] = intent
        return parsed


# %%
# ==================== 第八部分：专业Agent基类 ====================
class BaseAgent:
    @staticmethod
    def _prepare_messages(message: str, chat_history: List[Dict] = None, max_history: int = 6):
        if chat_history is None:
            chat_history = []
        messages = []
        for msg in chat_history[-max_history:]:
            role = msg["role"]
            if role in ("user", "human"):
                messages.append(HumanMessage(content=msg["content"]))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=msg["content"]))
            else:
                logger.warning(f"忽略未知消息角色: {role}")
                continue
        messages.append(HumanMessage(content=message))
        return messages


# %%
# ==================== 第九部分：各专业领域Agent ====================
class TechSupportAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [search_faq]
        self.system_prompt = """你是一个专业的技术支持工程师，你的职责是：
1.分析用户遇到的技术问题
2.提供清晰的故障排除步骤
3.使用search_faq工具查找相关解决方案
4.如果问题超出能力范围，建议升级到人工支持
回复要求：
- 语气友好专业
- 步骤清晰有序
- 提供多个可能解决方案"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，我暂时无法处理您的问题。建议联系人工客服"


class OrderServiceAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [query_order, track_shipping]
        self.system_prompt = """你是一个专业的订单服务专员。你的职责是：
1. 帮助用户查询订单状态
2. 提供物流跟踪信息
3. 解答退换货相关问题
4. 使用工具获取准确信息

回复要求：
- 信息准确完整
- 主动提供相关信息
- 如果需要订单号，礼貌询问"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，订单查询服务暂时不可用，请稍后再试。"


class ProductConsultAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [search_product, get_product_recommendations]
        self.system_prompt = """你是一个热情的产品顾问。你的职责是：
1. 介绍产品功能和特点
2. 根据用户需求推荐合适的产品
3. 解答价格和库存问题
4. 使用工具获取最新产品信息

回复要求：
- 热情有亲和力
- 突出产品优势
- 根据用户需求推荐
- 不要过度推销"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，产品信息查询暂时不可用。请稍后再试。"


class GeneralChatAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = []
        self.system_prompt = """你是一个友善、耐心、像人一样的智能客服助手。
你可以处理任何问题，包括：
- 闲聊：你好、今天天气、心情如何
- 功能询问：你能做什么、怎么使用
- 模糊问题：我不太清楚、怎么办
- 非业务问题：帮我写首诗、讲个笑话

回答要求：
- 像真正的客服人员一样自然、温和、体贴
- 当用户问你能做什么时，主动介绍自己的业务范围（订单查询、产品咨询、技术支持等）
- 如果实在无法回答，可以说“这个问题我暂时不太擅长，但你可以具体告诉我需要什么帮助吗？”
- 绝对不要直接建议转人工，除非用户明确要求或者问题涉及敏感内容

保持人性化的语气，不要像机器人一样死板。"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "嗯...我还在学习中，你能再说一遍吗？"


# %%
# ==================== 第十部分：质量检查器 ====================
class QualityChecker:
    def __init__(self):
        self.llm = get_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是客服质量检查专家。评估客服回复的质量。

评估维度：
1. 相关性（0-25分）：回复是否针对用户问题
2. 完整性（0-25分）：是否提供了足够的信息
3. 专业性（0-25分）：语言是否专业得体
4. 有用性（0-25分）：是否真正帮助到用户

返回格式（JSON）：
{{"total_score": 0-100, "needs_escalation": True/False, "reason": "评估说明"}}

只返回JSON。"""),
            ("human", """用户问题：{user_message}
客服回复：{agent_response}

请评估：""")
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm | StrOutputParser()
        result = llm_invoke_with_retry(chain, {"user_message": user_message, "agent_response": agent_response})
        default_result = {"total_score": 60, "needs_escalation": False, "reason": "评估完成"}
        return safe_parse_json(result, default_result)


# %%
# ==================== 第十一部分：客服系统主控类 ====================
class CustomerServiceSystem:
    INTENT_CONFIDENCE_THRESHOLD = 0.6
    QUALITY_SCORE_THRESHOLD = 0.6

    def __init__(self):
        self.classifier = IntentClassifier()
        self.tech_agent = TechSupportAgent()
        self.order_agent = OrderServiceAgent()
        self.product_agent = ProductConsultAgent()
        self.general_agent = GeneralChatAgent()
        self.quality_checker = QualityChecker()
        self.current_history = []
        self.graph = self._build_graph()

    def _build_graph(self):
        def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("分析用户意图")
            result = self.classifier.classify(state["user_message"])
            state["intent"] = result.get("intent", "general_chat")
            state["confidence"] = result.get("confidence", 0.3)
            logger.info(f"意图：{state['intent']} (置信度: {state['confidence']:.2f})")
            return state

        def route_to_agent(state: CustomerServiceState) -> Literal[
            "tech_support", "order_service", "product_consult", "general_chat", "escalate"]:
            intent = state["intent"]
            confidence = state["confidence"]
            if intent == "escalate" and confidence >= 0.7:
                return "escalate"
            if confidence < self.INTENT_CONFIDENCE_THRESHOLD:
                return "general_chat"
            if intent == "tech_support":
                return "tech_support"
            elif intent == "order_service":
                return "order_service"
            elif intent == "product_consult":
                return "product_consult"
            else:
                return "general_chat"

        def tech_support_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("技术支持代理处理中")
            response = self.tech_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def order_service_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("订单服务代理处理中")
            response = self.order_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def product_consult_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("产品咨询代理处理中")
            response = self.product_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def general_chat_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("通用对话代理处理中")
            response = self.general_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def escalate_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("升级到人工客服")
            state["needs_escalation"] = True
            state["escalation_reason"] = "用户明确要求人工服务"
            state["quality_score"] = 1.0
            state["agent_response"] = """非常抱歉，您的问题需要人工客服来处理。
我已经为您转接人工客服，请稍后...

在等待期间，你也可以：
1. 拨打客服热线：400-xxx-xxxx
2. 发送邮件至：support@example.com
3. 工作日 9:00-18:00 在线客服响应更快

感谢您的耐心等待！"""
            return state

        def quality_check(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("执行质量检查")
            result = self.quality_checker.check(state["user_message"], state["agent_response"])

            raw_score = result.get("total_score", 0)
            try:
                score = float(raw_score) / 100.0
            except (ValueError, TypeError):
                score = 0.6
            state["quality_score"] = score

            if state["intent"] == "general_chat":
                model_says_escalate = result.get("needs_escalation", False)
                if (model_says_escalate or state["quality_score"] < 0.3) and not state.get("already_escalated", False):
                    state["needs_escalation"] = True
                    state["escalation_reason"] = result.get("reason", "通用对话质量过低")
                    state["already_escalated"] = True
                else:
                    state["needs_escalation"] = False
            else:
                if (result.get("needs_escalation", False) or state["quality_score"] < self.QUALITY_SCORE_THRESHOLD) \
                        and not state.get("already_escalated", False):
                    state["needs_escalation"] = True
                    state["escalation_reason"] = result.get("reason", "质量检查未通过")
                    state["already_escalated"] = True

            logger.info(f"质量评分：{state['quality_score']:.2f}")
            return state

        def should_escalate(state: CustomerServiceState) -> Literal["escalate_final", "respond"]:
            if state.get("needs_escalation", False):
                return "escalate_final"
            return "respond"

        def final_escalate(state: CustomerServiceState) -> CustomerServiceState:
            original_response = state["agent_response"]
            state["agent_response"] = f"""{original_response}
系统提示：由于此问题可能需要更专业的处理，我们建议您联系人工客服以获得更好的服务。"""
            return state

        def respond(state: CustomerServiceState) -> CustomerServiceState:
            return state

        graph = StateGraph(CustomerServiceState)
        graph.add_node("classify", classify_intent)
        graph.add_node("tech_support", tech_support_handler)
        graph.add_node("order_service", order_service_handler)
        graph.add_node("product_consult", product_consult_handler)
        graph.add_node("general_chat", general_chat_handler)
        graph.add_node("escalate", escalate_handler)
        graph.add_node("quality_check", quality_check)
        graph.add_node("escalate_final", final_escalate)
        graph.add_node("respond", respond)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            route_to_agent,
            {
                "tech_support": "tech_support",
                "order_service": "order_service",
                "product_consult": "product_consult",
                "general_chat": "general_chat",
                "escalate": "escalate"
            }
        )
        graph.add_edge("tech_support", "quality_check")
        graph.add_edge("order_service", "quality_check")
        graph.add_edge("product_consult", "quality_check")
        graph.add_edge("general_chat", "quality_check")
        graph.add_edge("escalate", "respond")

        graph.add_conditional_edges(
            "quality_check",
            should_escalate,
            {
                "escalate_final": "escalate_final",
                "respond": "respond"
            }
        )
        graph.add_edge("escalate_final", END)
        graph.add_edge("respond", END)

        return graph.compile()

    def handle_message(self, message: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        try:
            logger.info(f"用户消息: {message}")
            initial_state = {
                "user_message": message,
                "chat_history": chat_history or [],
                "intent": "",
                "confidence": 0.0,
                "agent_response": "",
                "needs_escalation": False,
                "escalation_reason": "",
                "quality_score": 0.0,
                "already_escalated": False,
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            result = self.graph.invoke(initial_state)
            return {
                "response": result["agent_response"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "quality_score": result["quality_score"],
                "escalated": result["needs_escalation"]
            }
        except Exception as e:
            logger.error(f"处理消息时发生异常: {e}", exc_info=True)
            return {
                "response": "非常抱歉，系统暂时遇到了一点问题，请稍后再试或联系人工客服。",
                "intent": "general_chat",
                "confidence": 0.0,
                "quality_score": 0.0,
                "escalated": False
            }


# %%
# ==================== 第十二部分：演示主程序 ====================
def main():
    print("=" * 60)
    print("多代理智能客服系统演示")
    print("=" * 60)
    print("\n初始化客服系统...")
    system = CustomerServiceSystem()
    print("系统初始化完成！")

    test_cases = [
        {"category": "技术支持", "messages": ["我的蓝牙耳机连接不上手机怎么办？", "手表充电很慢，是不是坏了？"]},
        {"category": "订单服务", "messages": ["帮我查一下订单 ORD001 的物流状态", "我的订单什么时候能到？订单号是 ORD002"]},
        {"category": "产品咨询", "messages": ["你们有什么智能手表推荐吗？预算1500左右", "无线耳机有什么功能？"]},
        {"category": "人工升级", "messages": ["我要投诉！这是第三次出问题了！", "我想和你们经理谈谈"]}
    ]

    for test in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试类别: {test['category']}")
        print('=' * 60)
        chat_history = []
        for message in test["messages"]:
            result = system.handle_message(message, chat_history)
            print("\n客服回复:")
            print(f"{result['response']}")
            print("\n处理信息:")
            print(f"   - 意图: {result['intent']}")
            print(f"   - 置信度: {result['confidence']:.2f}")
            print(f"   - 质量评分: {result['quality_score']:.2f}")
            print(f"   - 是否升级: {'是' if result['escalated'] else '否'}")
            print("-" * 60)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": result['response']})

    print("\n" + "=" * 60)
    print("交互式对话演示")
    print("=" * 60)
    print("提示: 输入 'quit' 退出")
    chat_history = []
    while True:
        user_input = input("\n您: ").strip()
        if user_input.lower() == 'quit':
            print("\n感谢使用智能客服系统，再见！")
            break
        if not user_input:
            continue
        result = system.handle_message(user_input, chat_history)
        print(f"\n客服: {result['response']}")
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": result['response']})


if __name__ == "__main__":
    main()
# %%  (Jupyter Notebook 单元格标记，用于分隔代码块，不影响运行)
# ==================== 第一部分：导入必要的库 ====================
import os, sys, json, logging  # 导入操作系统接口、系统参数、JSON处理、日志记录库
from typing import List, Dict, Any, TypedDict, Literal  # 导入类型提示工具
from datetime import datetime  # 导入日期时间处理类

# 将项目根目录添加到系统路径，确保可以导入 shared 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import safe_parse_json, setup_logging, ModelCache, llm_invoke_with_retry  # 从共享模块导入工具函数

# 导入 LangChain 核心消息类型
from langchain_core.messages import HumanMessage, AIMessage
# 导入聊天提示模板
from langchain_core.prompts import ChatPromptTemplate
# 导入字符串输出解析器，用于提取模型回复的纯文本
from langchain_core.output_parsers import StrOutputParser
# 导入工具装饰器，用于将函数注册为 LangChain 工具
from langchain_core.tools import tool
# 导入 LangGraph 状态图及起始/结束常量
from langgraph.graph import StateGraph, START, END
# 导入创建 Agent 的工厂函数
from langchain.agents import create_agent

# %%
# ==================== 第二部分：配置日志与环境变量 ====================
logger = setup_logging(__name__)  # 初始化日志记录器
_model_cache = ModelCache(temperature=0.2, max_tokens=1000)  # 创建 LLM 缓存实例（温度 0.2，最大输出 1000 tokens）

def get_model():
    """获取共享的 LLM 模型实例"""
    return _model_cache.get()  # 从缓存中获取或创建模型

# %%
# ==================== 第三部分：模拟数据 ====================
# 模拟订单数据库，键为订单 ID，值为订单详情字典
MOCK_ORDERS = {
    "ORD001": {"status": "已发货", "product": "智能手表 Pro", "price": 1299, "shipping": "顺丰快递", "tracking": "SF1234567890", "estimated_delivery": "2024-12-20"},
    "ORD002": {"status": "处理中", "product": "无线耳机 Max", "price": 899, "shipping": "待发货", "tracking": None, "estimated_delivery": "2024-12-22"},
    "ORD003": {"status": "已完成", "product": "便携充电宝", "price": 199, "shipping": "已签收", "tracking": "YT9876543210", "estimated_delivery": "2024-12-15"}
}

# 模拟产品数据库，键为产品名称，值为产品属性字典
MOCK_PRODUCTS = {
    "智能手表 Pro": {"price": 1299, "features": ["心率监测", "GPS定位", "防水50米", "7天续航"], "stock": 50, "rating": 4.8},
    "无线耳机 Max": {"price": 899, "features": ["主动降噪", "40小时续航", "蓝牙5.3", "通话降噪"], "stock": 120, "rating": 4.6},
    "便携充电宝": {"price": 199, "features": ["20000mAh", "快充支持", "双USB输出", "LED显示"], "stock": 200, "rating": 4.5},
    "智能音箱": {"price": 499, "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi音质"], "stock": 80, "rating": 4.7}
}

# 常见问题数据库，键为问题分类，值为标准答案
FAQ_DATABASE = {
    "连接问题": "请尝试以下步骤：1) 重启设备 2) 检查蓝牙是否开启 3) 删除配对记录后重新配对 4) 确保设备电量充足",
    "充电问题": "建议使用原装充电器，检查充电线是否损坏。如果问题持续，可能需要更换电池或送修。",
    "软件更新": "打开设备对应的APP，进入设置-关于-检查更新，按提示操作即可完成更新。",
    "退货政策": "我们支持7天无理由退货，30天内有质量问题可换货。请保留好购买凭证和完整包装。"
}

# %%
# ==================== 第四部分：工具函数 ====================
@tool  # 将函数注册为 LangChain 工具，Agent 可以调用它
def query_order(order_id: str) -> str:
    """查询订单信息，根据订单号返回订单详情JSON字符串"""
    order = MOCK_ORDERS.get(order_id.upper())  # 获取订单（将输入 ID 转为大写后查找）
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)  # 返回格式化的 JSON 字符串
    return f"未找到订单{order_id}"  # 未找到时返回提示信息

@tool
def track_shipping(tracking_number: str) -> str:
    """查询物流信息，根据快递单号返回物流状态描述"""
    if tracking_number.startswith("SF"):  # 顺丰快递
        return f"顺丰快递{tracking_number}:包裹已到达配送站，预计今日送达"
    elif tracking_number.startswith("YT"):  # 圆通快递
        return f"圆通快递{tracking_number}:已签收"
    return f"未找到物流信息{tracking_number}"  # 其他情况返回未找到

@tool
def search_product(keyword: str) -> str:
    """搜索产品信息，根据关键词在产品名称中匹配，返回匹配产品列表JSON"""
    results = []
    for name, info in MOCK_PRODUCTS.items():  # 遍历产品数据库
        if keyword.lower() in name.lower():  # 不区分大小写匹配产品名称
            results.append({"name": name, "price": info["price"], "features": info["features"], "rating": info["rating"]})
    if results:
        return json.dumps(results, ensure_ascii=False, indent=2)  # 返回 JSON 列表
    return f"未找到包含{keyword}的产品"

@tool
def get_product_recommendations(budget: int) -> str:
    """根据预算推荐产品，返回价格不超过预算且评分最高的前3款产品JSON"""
    recommendations = []
    for name, info in MOCK_PRODUCTS.items():
        if info['price'] <= budget:  # 价格在预算内
            recommendations.append({"name": name, "price": info['price'], "rating": info['rating']})
    recommendations.sort(key=lambda x: x["price"], reverse=True)  # 按价格降序排列
    if recommendations:
        return json.dumps(recommendations[:3], ensure_ascii=False, indent=2)  # 返回前 3 个
    return f"在预算{budget}内暂无推荐产品"

@tool
def search_faq(problem_type: str) -> str:
    """搜索常见问题解答，根据问题类型关键词匹配FAQ答案"""
    for key, answer in FAQ_DATABASE.items():
        if problem_type in key:  # 如果用户描述的关键词出现在问题分类中
            return f"【{key}】\n{answer}"  # 返回格式化答案
    return "未找到相关FAQ，建议联系人工客服获取更多帮助。"

# %%
# ==================== 第五部分：客服系统状态定义 ====================
class CustomerServiceState(TypedDict):
    """定义 LangGraph 工作流中传递的状态字典的结构"""
    user_message: str               # 用户原始消息
    chat_history: List[Dict[str, str]]  # 对话历史
    intent: str                     # 分类后的意图
    confidence: float               # 意图置信度
    agent_response: str            # Agent 生成的回复
    needs_escalation: bool          # 是否需要升级到人工
    escalation_reason: str         # 升级原因
    quality_score: float           # 质检评分（0~1）
    already_escalated: bool        # 是否已经触发过升级（防止重复）
    metadata: Dict[str, Any]       # 额外元数据（如时间戳）

# %%
# ==================== 第六部分：安全JSON解析工具 ====================
# safe_parse_json 已从 shared.utils 导入，此处注释说明

# %%
# ==================== 第七部分：意图分类器 ====================
class IntentClassifier:
    """使用 LLM 对用户消息进行意图分类"""
    VALID_INTENTS = {"tech_support", "order_service", "product_consult", "general_chat", "escalate"}  # 有效意图集合

    def __init__(self):
        self.llm = get_model()  # 获取语言模型实例
        self.prompt = ChatPromptTemplate.from_messages([  # 定义提示模板
            ("system", """你是一个意图分类专家，分析用户消息并返回意图分类。
            
可选意图：
- tech_support: 具体技术问题、故障排除、使用帮助（如“蓝牙连不上”、“充电慢”）
- order_service: 具体订单查询、物流跟踪、退换货（如“查订单 ORD001”、“快递到哪了”）
- product_consult: 具体产品咨询、价格询问、功能介绍（如“智能手表多少钱”、“推荐一款耳机”）
- general_chat: 通用对话、闲聊、功能询问、模糊问题、非业务问题（如“你好”、“你能做什么”、“帮我写首诗”、“我不懂”）
- escalate: 明确要求人工客服、投诉、严重不满、要求经理、连续无法解决问题（如“我要投诉”、“转人工”、“叫你们经理来”）

返回格式(JSON):
{{"intent": "意图类型","confidence": 0.0-1.0, "reason": "分类原因"}}

只返回JSON，不要其他内容。"""),
            ("human", "{message}")  # 用户消息占位符
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        """对用户消息进行分类，返回意图、置信度等信息"""
        chain = self.prompt | self.llm | StrOutputParser()  # 构建链：提示 -> LLM -> 字符串解析
        result = llm_invoke_with_retry(chain, {"message": message})  # 调用链并获取结果
        default_result = {"intent": "general_chat", "confidence": 0.5, "reason": "解析失败"}  # 默认结果（解析失败时使用）
        parsed = safe_parse_json(result, default_result)  # 安全解析 LLM 返回的 JSON
        intent = parsed.get("intent", "general_chat")  # 提取意图
        if intent not in self.VALID_INTENTS:  # 如果意图不在有效集合中
            intent = "general_chat"  # 降级为通用对话
        parsed["intent"] = intent  # 更新意图字段
        return parsed

# %%
# ==================== 第八部分：专业Agent基类 ====================
class BaseAgent:
    """所有专业 Agent 的基类，提供消息预处理功能"""
    @staticmethod
    def _prepare_messages(message: str, chat_history: List[Dict] = None, max_history: int = 6):
        """将用户消息和对话历史转换为 LangChain 消息列表"""
        if chat_history is None:
            chat_history = []  # 如果没有历史，使用空列表
        messages = []
        for msg in chat_history[-max_history:]:  # 只取最近 max_history 条历史，避免上下文过长
            role = msg["role"]
            if role in ("user", "human"):  # 用户消息
                messages.append(HumanMessage(content=msg["content"]))
            elif role in ("assistant", "ai"):  # AI 回复
                messages.append(AIMessage(content=msg["content"]))
            else:
                logger.warning(f"忽略未知消息角色: {role}")  # 记录未知角色警告
                continue
        messages.append(HumanMessage(content=message))  # 最后添加当前用户消息
        return messages

# %%
# ==================== 第九部分：各专业领域Agent ====================
class TechSupportAgent(BaseAgent):
    """技术支持 Agent，使用 search_faq 工具"""
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [search_faq]  # 绑定 FAQ 搜索工具
        self.system_prompt = """你是一个专业的技术支持工程师，你的职责是：
1.分析用户遇到的技术问题
2.提供清晰的故障排除步骤
3.使用search_faq工具查找相关解决方案
4.如果问题超出能力范围，建议升级到人工支持
回复要求：
- 语气友好专业
- 步骤清晰有序
- 提供多个可能解决方案"""
        self.agent = create_agent(  # 创建 LangChain Agent
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)  # 预处理消息
        result = llm_invoke_with_retry(self.agent, {"messages": messages})  # 调用 Agent
        if result.get("messages"):  # 如果返回了消息列表
            return result["messages"][-1].content  # 返回最后一条消息的内容
        return "抱歉，我暂时无法处理您的问题。建议联系人工客服"  # 兜底回复

class OrderServiceAgent(BaseAgent):
    """订单服务 Agent，使用订单查询和物流跟踪工具"""
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [query_order, track_shipping]
        self.system_prompt = """你是一个专业的订单服务专员。你的职责是：
1. 帮助用户查询订单状态
2. 提供物流跟踪信息
3. 解答退换货相关问题
4. 使用工具获取准确信息

回复要求：
- 信息准确完整
- 主动提供相关信息
- 如果需要订单号，礼貌询问"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，订单查询服务暂时不可用，请稍后再试。"

class ProductConsultAgent(BaseAgent):
    """产品咨询 Agent，使用产品搜索和推荐工具"""
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [search_product, get_product_recommendations]
        self.system_prompt = """你是一个热情的产品顾问。你的职责是：
1. 介绍产品功能和特点
2. 根据用户需求推荐合适的产品
3. 解答价格和库存问题
4. 使用工具获取最新产品信息

回复要求：
- 热情有亲和力
- 突出产品优势
- 根据用户需求推荐
- 不要过度推销"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，产品信息查询暂时不可用。请稍后再试。"

class GeneralChatAgent(BaseAgent):
    """通用对话 Agent，处理闲聊和模糊问题，无工具依赖"""
    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = []  # 通用聊天不使用工具
        self.system_prompt = """你是一个友善、耐心、像人一样的智能客服助手。
你可以处理任何问题，包括：
- 闲聊：你好、今天天气、心情如何
- 功能询问：你能做什么、怎么使用
- 模糊问题：我不太清楚、怎么办
- 非业务问题：帮我写首诗、讲个笑话

回答要求：
- 像真正的客服人员一样自然、温和、体贴
- 当用户问你能做什么时，主动介绍自己的业务范围（订单查询、产品咨询、技术支持等）
- 如果实在无法回答，可以说“这个问题我暂时不太擅长，但你可以具体告诉我需要什么帮助吗？”
- 绝对不要直接建议转人工，除非用户明确要求或者问题涉及敏感内容

保持人性化的语气，不要像机器人一样死板。"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = llm_invoke_with_retry(self.agent, {"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "嗯...我还在学习中，你能再说一遍吗？"

# %%
# ==================== 第十部分：质量检查器 ====================
class QualityChecker:
    """使用 LLM 评估客服回复的质量"""
    def __init__(self):
        self.llm = get_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是客服质量检查专家。评估客服回复的质量。

评估维度：
1. 相关性（0-25分）：回复是否针对用户问题
2. 完整性（0-25分）：是否提供了足够的信息
3. 专业性（0-25分）：语言是否专业得体
4. 有用性（0-25分）：是否真正帮助到用户

返回格式（JSON）：
{{"total_score": 0-100, "needs_escalation": True/False, "reason": "评估说明"}}

只返回JSON。"""),
            ("human", """用户问题：{user_message}
客服回复：{agent_response}

请评估：""")
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        """执行质量检查，返回评分和是否需升级的建议"""
        chain = self.prompt | self.llm | StrOutputParser()
        result = llm_invoke_with_retry(chain, {"user_message": user_message, "agent_response": agent_response})
        default_result = {"total_score": 60, "needs_escalation": False, "reason": "评估完成"}  # 默认中等评分
        return safe_parse_json(result, default_result)

# %%
# ==================== 第十一部分：客服系统主控类 ====================
class CustomerServiceSystem:
    """多代理智能客服系统，使用 LangGraph 编排工作流"""
    INTENT_CONFIDENCE_THRESHOLD = 0.6  # 意图置信度阈值，低于此值降级为通用对话
    QUALITY_SCORE_THRESHOLD = 0.6      # 质检评分阈值，低于此值建议升级

    def __init__(self):
        # 初始化各组件
        self.classifier = IntentClassifier()
        self.tech_agent = TechSupportAgent()
        self.order_agent = OrderServiceAgent()
        self.product_agent = ProductConsultAgent()
        self.general_agent = GeneralChatAgent()
        self.quality_checker = QualityChecker()
        self.current_history = []  # 当前对话历史（由外部 API 层管理，此处供参考）
        self.graph = self._build_graph()  # 构建并编译状态图

    def _build_graph(self):
        """构建 LangGraph 工作流：意图分类 -> 路由 -> Agent处理 -> 质检 -> 可能升级"""
        
        # --- 节点定义 ---
        def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
            """意图分类节点"""
            logger.info("分析用户意图")
            result = self.classifier.classify(state["user_message"])
            state["intent"] = result.get("intent", "general_chat")
            state["confidence"] = result.get("confidence", 0.3)
            logger.info(f"意图：{state['intent']} (置信度: {state['confidence']:.2f})")
            return state

        def route_to_agent(state: CustomerServiceState) -> Literal[
            "tech_support", "order_service", "product_consult", "general_chat", "escalate"]:
            """根据意图和置信度决定下一个节点"""
            intent = state["intent"]
            confidence = state["confidence"]
            if intent == "escalate" and confidence >= 0.7:  # 明确要求且置信度高，直接转人工
                return "escalate"
            if confidence < self.INTENT_CONFIDENCE_THRESHOLD:  # 置信度低，降级为通用对话
                return "general_chat"
            # 正常路由到对应的专业 Agent
            if intent == "tech_support":
                return "tech_support"
            elif intent == "order_service":
                return "order_service"
            elif intent == "product_consult":
                return "product_consult"
            else:
                return "general_chat"

        def tech_support_handler(state: CustomerServiceState) -> CustomerServiceState:
            """技术支持处理节点"""
            logger.info("技术支持代理处理中")
            response = self.tech_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def order_service_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("订单服务代理处理中")
            response = self.order_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def product_consult_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("产品咨询代理处理中")
            response = self.product_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def general_chat_handler(state: CustomerServiceState) -> CustomerServiceState:
            logger.info("通用对话代理处理中")
            response = self.general_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def escalate_handler(state: CustomerServiceState) -> CustomerServiceState:
            """直接升级处理节点（用户明确要求人工）"""
            logger.info("升级到人工客服")
            state["needs_escalation"] = True
            state["escalation_reason"] = "用户明确要求人工服务"
            state["quality_score"] = 1.0  # 直接升级时评分为满分（因为就是用户想要的）
            state["agent_response"] = """非常抱歉，您的问题需要人工客服来处理。
我已经为您转接人工客服，请稍后...

在等待期间，你也可以：
1. 拨打客服热线：400-xxx-xxxx
2. 发送邮件至：support@example.com
3. 工作日 9:00-18:00 在线客服响应更快

感谢您的耐心等待！"""
            return state

        def quality_check(state: CustomerServiceState) -> CustomerServiceState:
            """质量检查节点，决定是否需要建议升级"""
            logger.info("执行质量检查")
            result = self.quality_checker.check(state["user_message"], state["agent_response"])

            raw_score = result.get("total_score", 0)
            try:
                score = float(raw_score) / 100.0  # 转换为 0~1 的小数
            except (ValueError, TypeError):
                score = 0.6  # 转换失败时给默认中等分
            state["quality_score"] = score

            # 根据意图类型采用不同的升级策略
            if state["intent"] == "general_chat":
                # 通用对话仅当明确要求升级或质量极低时才升级（阈值 0.3）
                model_says_escalate = result.get("needs_escalation", False)
                if (model_says_escalate or state["quality_score"] < 0.3) and not state.get("already_escalated", False):
                    state["needs_escalation"] = True
                    state["escalation_reason"] = result.get("reason", "通用对话质量过低")
                    state["already_escalated"] = True
                else:
                    state["needs_escalation"] = False
            else:
                # 专业领域：模型判断需升级或分数低于阈值，且未重复升级
                if (result.get("needs_escalation", False) or state["quality_score"] < self.QUALITY_SCORE_THRESHOLD) \
                        and not state.get("already_escalated", False):
                    state["needs_escalation"] = True
                    state["escalation_reason"] = result.get("reason", "质量检查未通过")
                    state["already_escalated"] = True

            logger.info(f"质量评分：{state['quality_score']:.2f}")
            return state

        def should_escalate(state: CustomerServiceState) -> Literal["escalate_final", "respond"]:
            """条件边：根据 needs_escalation 决定下一跳"""
            if state.get("needs_escalation", False):
                return "escalate_final"
            return "respond"

        def final_escalate(state: CustomerServiceState) -> CustomerServiceState:
            """最终升级节点：在原始回复后附加升级提示"""
            original_response = state["agent_response"]
            state["agent_response"] = f"""{original_response}
系统提示：由于此问题可能需要更专业的处理，我们建议您联系人工客服以获得更好的服务。"""
            return state

        def respond(state: CustomerServiceState) -> CustomerServiceState:
            """直接输出回复的节点（不升级）"""
            return state

        # --- 图构建 ---
        graph = StateGraph(CustomerServiceState)
        graph.add_node("classify", classify_intent)
        graph.add_node("tech_support", tech_support_handler)
        graph.add_node("order_service", order_service_handler)
        graph.add_node("product_consult", product_consult_handler)
        graph.add_node("general_chat", general_chat_handler)
        graph.add_node("escalate", escalate_handler)  # 直接升级节点
        graph.add_node("quality_check", quality_check)
        graph.add_node("escalate_final", final_escalate)
        graph.add_node("respond", respond)

        graph.add_edge(START, "classify")  # 开始 -> 意图分类
        graph.add_conditional_edges(  # 分类后的条件路由
            "classify",
            route_to_agent,
            {
                "tech_support": "tech_support",
                "order_service": "order_service",
                "product_consult": "product_consult",
                "general_chat": "general_chat",
                "escalate": "escalate"
            }
        )
        # 所有专业 Agent 和通用 Agent 处理完后进入质检
        graph.add_edge("tech_support", "quality_check")
        graph.add_edge("order_service", "quality_check")
        graph.add_edge("product_consult", "quality_check")
        graph.add_edge("general_chat", "quality_check")
        graph.add_edge("escalate", "respond")  # 直接升级后跳过质检，直接回复

        graph.add_conditional_edges(  # 质检后的条件路由
            "quality_check",
            should_escalate,
            {
                "escalate_final": "escalate_final",
                "respond": "respond"
            }
        )
        graph.add_edge("escalate_final", END)
        graph.add_edge("respond", END)

        return graph.compile()  # 编译为可执行的应用

    def handle_message(self, message: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """处理用户消息的入口函数，返回包含回复、意图、评分等的字典"""
        try:
            logger.info(f"用户消息: {message}")
            initial_state = {  # 构建初始状态
                "user_message": message,
                "chat_history": chat_history or [],
                "intent": "",
                "confidence": 0.0,
                "agent_response": "",
                "needs_escalation": False,
                "escalation_reason": "",
                "quality_score": 0.0,
                "already_escalated": False,
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            result = self.graph.invoke(initial_state)  # 运行工作流
            return {
                "response": result["agent_response"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "quality_score": result["quality_score"],
                "escalated": result["needs_escalation"]
            }
        except Exception as e:
            logger.error(f"处理消息时发生异常: {e}", exc_info=True)
            # 系统异常时返回友好的兜底回复，不触发升级
            return {
                "response": "非常抱歉，系统暂时遇到了一点问题，请稍后再试或联系人工客服。",
                "intent": "general_chat",
                "confidence": 0.0,
                "quality_score": 0.0,
                "escalated": False
            }

# %%
# ==================== 第十二部分：演示主程序 ====================
def main():
    """命令行演示：运行预设测试用例和交互式对话"""
    print("=" * 60)
    print("多代理智能客服系统演示")
    print("=" * 60)
    print("\n初始化客服系统...")
    system = CustomerServiceSystem()  # 创建系统实例
    print("系统初始化完成！")

    # 预定义测试用例
    test_cases = [
        {"category": "技术支持", "messages": ["我的蓝牙耳机连接不上手机怎么办？", "手表充电很慢，是不是坏了？"]},
        {"category": "订单服务", "messages": ["帮我查一下订单 ORD001 的物流状态", "我的订单什么时候能到？订单号是 ORD002"]},
        {"category": "产品咨询", "messages": ["你们有什么智能手表推荐吗？预算1500左右", "无线耳机有什么功能？"]},
        {"category": "人工升级", "messages": ["我要投诉！这是第三次出问题了！", "我想和你们经理谈谈"]}
    ]

    for test in test_cases:  # 遍历每个测试类别
        print(f"\n{'=' * 60}")
        print(f"测试类别: {test['category']}")
        print('=' * 60)
        chat_history = []
        for message in test["messages"]:
            result = system.handle_message(message, chat_history)  # 处理消息
            print("\n客服回复:")
            print(f"{result['response']}")
            print("\n处理信息:")
            print(f"   - 意图: {result['intent']}")
            print(f"   - 置信度: {result['confidence']:.2f}")
            print(f"   - 质量评分: {result['quality_score']:.2f}")
            print(f"   - 是否升级: {'是' if result['escalated'] else '否'}")
            print("-" * 60)
            # 更新对话历史
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": result['response']})

    # 交互式对话模式
    print("\n" + "=" * 60)
    print("交互式对话演示")
    print("=" * 60)
    print("提示: 输入 'quit' 退出")
    chat_history = []
    while True:
        user_input = input("\n您: ").strip()
        if user_input.lower() == 'quit':  # 输入 quit 退出
            print("\n感谢使用智能客服系统，再见！")
            break
        if not user_input:  # 忽略空输入
            continue
        result = system.handle_message(user_input, chat_history)
        print(f"\n客服: {result['response']}")
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": result['response']})

if __name__ == "__main__":
    main()  # 执行主函数
