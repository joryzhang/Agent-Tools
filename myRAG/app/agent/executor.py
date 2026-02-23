"""
RAG Agent Executor - 基于 LangGraph 的智能体执行器

架构:
  Router → retrieve (纯检索, knowledge/mixed 意图)
         → agent (统一 LLM 生成, 所有意图)
         → tools → agent → END

设计原则:
- RAG retrieve 节点只做检索 + 格式化，不调 LLM
- Agent 节点是唯一的 LLM 生成出口
- 检索结果通过 state.context 传递, 作为 SystemMessage 注入 Agent
- mixed 意图: 先检索文档 → agent 可同时使用文档 + 工具
- Chat History: 支持多轮对话上下文
"""

from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import logging
import asyncio

from app.tools.user_info_tool import get_user_info, get_user_stats
from app.tools.rag_tool import EnterpriseRAGTool
from app.agent.prompts import LANGGRAPH_SYSTEM_PROMPT, RAG_CONTEXT_SYSTEM_TEMPLATE
from app.rag.utils import format_docs, truncate_context
from app.config import settings

logger = logging.getLogger(__name__)

# HybridRouter 单例 (所有 RAGAgent 实例共享)
_shared_router: 'HybridRouter | None' = None


def _get_shared_router():
    """获取或创建共享路由器 (首次调用时初始化)"""
    global _shared_router
    if _shared_router is None:
        from app.agent.router import HybridRouter
        from models import get_ali_model_client
        from app.config import settings
        _shared_router = HybridRouter(
            llm=get_ali_model_client(temperature=0.1),
            threshold=settings.SEMANTIC_THRESHOLD,
        )
        logger.info("[RAGAgent] 共享 HybridRouter 创建完成")
    return _shared_router


async def warmup_router():
    """
    预热共享路由器的 SemanticRouter 向量缓存

    应在应用 startup 时调用, 避免首次请求承受初始化开销。
    """
    router = _get_shared_router()
    await router.semantic_router.initialize()
    logger.info("[RAGAgent] SemanticRouter 预热完成")


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: int
    intent_result: dict
    context: str  # RAG 检索到的文档上下文 (由 retrieve 节点填充)
    temperature: float  # 请求级温度, RAG 路径强制为 0


class RAGAgent:
    """
    RAG Agent 执行器

    统一架构:
    - retrieve 节点: 纯检索 + format_docs → 注入 state.context
    - agent 节点: 唯一 LLM 生成出口, 感知 context + chat history
    - tools 节点: 执行工具调用
    """

    def __init__(self, user_id: int, llm=None):
        self.user_id = user_id

        # 初始化 LLM
        from models import get_ali_model_client
        if llm is None:
            self.llm = get_ali_model_client()
        else:
            self.llm = llm

        # 初始化 RAG 检索器 (单例)
        from app.rag.retriever import get_retriever
        self.retriever = get_retriever()

        # 初始化工具
        self.tools = [
            get_user_info,       # @tool (StructuredTool)
            get_user_stats,      # @tool (StructuredTool)
            # EnterpriseRAGTool: Agent 的自主检索安全网
            # 当路由判定为 user_info/normal 但 Agent 在回答过程中发现需要知识库信息时,
            # 可以主动调用此工具。与 retrieve 节点功能相同但触发机制不同:
            # - retrieve 节点: 路由驱动 (knowledge/mixed 意图自动触发)
            # - RAGTool: Agent 自主驱动 (LLM 判断需要时主动调用)
            EnterpriseRAGTool(retriever=self.retriever),
        ]

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 预构建 System Prompt (避免每次 _call_model 重复 format)
        self._system_prompt = SystemMessage(
            content=LANGGRAPH_SYSTEM_PROMPT.format(user_id=self.user_id)
        )

        # 混合路由器 (全局单例, 所有 Agent 共享)
        self.intent_router = _get_shared_router()

        # 构建 LangGraph
        self.graph = self._build_graph()

        logger.info(f"[RAGAgent] 初始化完成, 用户ID: {user_id}")



    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 工作流

        Router → retrieve (knowledge/mixed) → agent → tools → agent → END
               → agent (user_info/normal)   → tools → agent → END
        """
        workflow = StateGraph(AgentState)

        # 定义节点
        workflow.add_node("router", self._route_intent)
        workflow.add_node("retrieve", self._retrieve)    # 纯检索节点
        workflow.add_node("agent", self._call_model)     # 统一 LLM 生成
        workflow.add_node("tools", self._call_tools)

        # 入口
        workflow.set_entry_point("router")

        # Router 条件边
        workflow.add_conditional_edges(
            "router",
            self._decide_route,
            {
                "retrieve": "retrieve",  # knowledge/mixed → 先检索
                "agent": "agent",        # user_info/normal → 直接 agent
            }
        )

        # 检索后 → agent
        workflow.add_edge("retrieve", "agent")

        # Agent 条件边
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )

        # 工具执行后 → agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # =========================================================
    # Graph 节点
    # =========================================================

    def _route_intent(self, state: AgentState) -> AgentState:
        """意图识别路由器 (不修改状态, 只用于路由决策)"""
        return state

    def _decide_route(self, state: AgentState) -> str:
        """
        决定路由方向

        knowledge / mixed → retrieve (需要检索文档)
        user_info / normal → agent (不需要检索, 直接对话或调工具)
        """
        intent_result = state.get("intent_result", {})
        intent = intent_result.get("intent", "normal")

        logger.info(f"[RAGAgent] 路由决策: {intent}")

        if intent in ("knowledge", "mixed"):
            return "retrieve"
        # user_info → agent 调用 get_user_info/get_user_stats 工具
        # normal → agent 直接对话
        return "agent"

    async def _retrieve(self, state: AgentState) -> AgentState:
        """
        纯检索节点 — 只负责检索和格式化, 不调用 LLM

        职责:
        1. 从向量库 + BM25 混合检索
        2. 回溯父文档 (含 override)
        3. 格式化文档
        4. 将结果注入 state.context
        """
        messages = state["messages"]
        # 取最后一条 HumanMessage 作为检索 query (支持多轮对话)
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            logger.warning("[RAGAgent] 未找到 HumanMessage, 跳过检索")
            return {"context": ""}

        logger.info(f"[RAGAgent] 执行纯检索: {query}")

        try:
            # 异步包装同步 IO
            docs = await asyncio.to_thread(
                self.retriever.get_relevant_documents, query
            )

            # 格式化文档
            context = format_docs(docs)

            # Token 预算截断: 防止 RAG 上下文溢出模型 context window
            context = truncate_context(context, max_tokens=settings.RAG_CONTEXT_MAX_TOKENS)

            logger.info(f"[RAGAgent] 检索完成, 上下文长度: {len(context)}")
            return {"context": context}

        except Exception as e:
            logger.error(f"[RAGAgent] 检索失败: {e}", exc_info=True)
            return {"context": ""}

    async def _call_model(self, state: AgentState) -> AgentState:
        """
        统一 LLM 生成节点

        每次调用都注入 System Prompt (LangGraph 标准模式):
        - System Prompt 预构建在 self._system_prompt, 不需要重复 format
        - RAG 上下文来自 state.context (由 retrieve 节点填充)
        - 支持 chat history (messages 中已含历史消息)
        """
        messages = list(state["messages"])
        context = state.get("context", "")
        req_temperature = state.get("temperature", 0.7)

        # 温度策略: RAG 路径强制 0 (确保忠实检索结果), 其他路径使用用户设定值
        temperature = 0.0 if context else req_temperature

        # 构建消息链: system prompt + (context) + 对话消息
        full_messages = [self._system_prompt]

        if context:
            full_messages.append(
                SystemMessage(content=RAG_CONTEXT_SYSTEM_TEMPLATE.format(context=context))
            )

        full_messages.extend(messages)

        logger.info(
            f"[RAGAgent] 调用模型, 消息数: {len(full_messages)}, "
            f"有检索上下文: {bool(context)}, 温度: {temperature}"
        )

        # 调用 LLM (动态绑定温度 + 工具)
        response = await self.llm_with_tools.bind(temperature=temperature).ainvoke(full_messages)

        return {"messages": [response]}

    async def _call_tools(self, state: AgentState) -> AgentState:
        """调用工具节点 (自定义, 注入 user_id)"""
        messages = state["messages"]
        last_message = messages[-1]
        user_id = state["user_id"]

        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            logger.info(f"[RAGAgent] 调用工具: {tool_name}, 参数: {tool_args}")

            # 注入 user_id
            if tool_name in ["get_user_info", "get_user_stats"]:
                tool_args["user_id"] = user_id
                logger.info(f"[RAGAgent] 注入 user_id={user_id}")

            # 查找并执行工具
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    result = await asyncio.to_thread(tool._run, **tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
                except Exception as e:
                    logger.error(f"[RAGAgent] 工具执行失败: {e}")
                    tool_messages.append(
                        ToolMessage(
                            content=f"工具执行失败: {str(e)}",
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
            else:
                logger.error(f"[RAGAgent] 工具未找到: {tool_name}")
                tool_messages.append(
                    ToolMessage(
                        content=f"工具未找到: {tool_name}",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )

        return {"messages": tool_messages}

    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续执行工具"""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(
                f"[RAGAgent] 需要调用工具: "
                f"{[tc['name'] for tc in last_message.tool_calls]}"
            )
            return "continue"

        logger.info("[RAGAgent] 生成最终答案, 结束")
        return "end"

    # =========================================================
    # 外部接口
    # =========================================================

    async def astream(self, query: str, chat_history: Optional[List[dict]] = None, temperature: float = 0.7):
        """
        执行 Agent (流式异步)

        Args:
            query: 当前用户问题
            chat_history: 对话历史, 格式: [{"role": "user", "content": "..."}, ...]
            temperature: LLM 生成温度, RAG 路径会被强制覆盖为 0

        Yields:
            dict: {"type": "intent"|"content"|"error", "data": ...}
        """
        try:
            logger.info(f"[RAGAgent] 开始流式执行: {query}")

            # 1. 意图识别 (只用当前 query, 不需要历史)
            try:
                intent_result = await self.intent_router.classify(query)

                logger.info(f"[RAGAgent] 意图识别: {intent_result.intent}")

                yield {
                    "type": "intent",
                    "data": {
                        "intent": intent_result.intent,
                        "confidence": intent_result.confidence,
                        "reasoning": intent_result.reasoning,
                        "source": intent_result.source,
                    }
                }

            except Exception as e:
                logger.error(f"[RAGAgent] 意图识别失败: {e}", exc_info=True)
                intent_result = type('obj', (object,), {
                    'intent': 'knowledge', 'source': 'fallback'
                })()

            # 2. 构建消息链 (chat history + 当前 query)
            messages: List[BaseMessage] = []

            if chat_history:
                for msg in chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

                logger.info(f"[RAGAgent] 注入 {len(messages)} 条历史消息")

            # 当前 query
            messages.append(HumanMessage(content=query))

            # 2.5 Token 截断: 保留最近的消息, 防止 context window 溢出
            if len(messages) > 1:  # 只有存在历史时才需要截断
                messages = trim_messages(
                    messages,
                    max_tokens=4000,        # 历史消息的 token 预算 根据模型可以调高
                    token_counter=self.llm,  # 用 LLM 的 tokenizer 计数
                    strategy="last",         # 保留最近的, 丢弃最旧的
                    start_on="human",        # 截断后第一条必须是 HumanMessage
                    include_system=False,    # 不计入 system message
                )
                logger.info(f"[RAGAgent] Token 截断后消息数: {len(messages)}")

            # 3. 构建初始状态
            initial_state = {
                "messages": messages,
                "user_id": self.user_id,
                "intent_result": {
                    "intent": intent_result.intent
                },
                "context": "",  # 初始为空, 由 retrieve 节点填充
                "temperature": temperature,  # 请求级温度
            }

            # 4. 流式执行 graph
            async for event in self.graph.astream_events(initial_state, version="v1"):
                kind = event["event"]

                # 监听 LLM 的流式 token
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield {
                            "type": "content",
                            "data": chunk.content
                        }

            logger.info("[RAGAgent] 流式执行完成")

        except Exception as e:
            logger.error(f"[RAGAgent] 执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "data": str(e)
            }
