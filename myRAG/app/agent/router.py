"""
Intent Router - 混合意图路由器

两层架构:
  L1: SemanticRouter (向量相似度, ~50ms)
  L2: LLM Router (兜底, ~1-2s, 结构化输出)

SemanticRouter 做快速比对 + 分数阈值过滤,
LLM Router 处理未匹配到的模糊/新奇表达。
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
import logging

from app.agent.semantic_router import SemanticRouter, SemanticClassification

logger = logging.getLogger(__name__)


# =========================================================
# 数据模型
# =========================================================

class LLMIntentResult(BaseModel):
    """LLM 结构化输出模型 — 直接由 LLM 返回，不需要正则解析"""
    intent: Literal["user_info", "knowledge", "mixed", "normal"] = Field(
        description="意图类型: user_info=用户信息查询, knowledge=知识库查询, mixed=混合查询, normal=普通闲聊"
    )
    confidence: float = Field(
        description="置信度 (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="分类理由 (简短说明)"
    )


class IntentClassification(BaseModel):
    """最终意图分类结果 (统一输出)"""
    intent: Literal["user_info", "knowledge", "mixed", "normal"] = Field(
        description="用户意图类型"
    )
    confidence: float = Field(
        description="置信度 (0-1)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="分类理由"
    )
    source: str = Field(
        default="llm",
        description="分类来源: semantic (向量匹配) 或 llm (LLM兜底)"
    )


# =========================================================
# L2: LLM Router (兜底, 结构化输出)
# =========================================================

LLM_CLASSIFY_PROMPT = """你是一个意图分类专家。请分析用户的问题，判断属于以下哪种类型:

1. user_info - 用户信息查询
   - 询问个人信息、账户余额、VIP等级、订单数量、消费金额等
   - 示例: "我的余额是多少?", "我是VIP吗?", "我消费了多少?"

2. knowledge - 知识库查询
   - 询问业务规则、流程、政策、产品说明等
   - 示例: "违约金怎么算?", "VIP有什么特权?", "如何申请退款?"

3. mixed - 混合查询
   - 同时涉及个人信息和知识库内容
   - 示例: "我是VIP吗?VIP有什么特权?", "我的余额够支付违约金吗?"

4. normal - 普通问题
   - 既不包含个人信息、也不包含知识库内容
   - 示例: "西安在哪里?", "你好", "今天天气怎么样?"

用户问题: {query}

请以 json 格式返回分类结果。"""


class LLMRouter:
    """
    基于 LLM 结构化输出的意图分类器 (兜底层)

    使用 with_structured_output 直接返回 Pydantic 对象,
    彻底消除正则解析。
    """

    def __init__(self, llm=None):
        if llm is None:
            from models import get_ali_model_client
            llm = get_ali_model_client(temperature=0.1)

        # 绑定结构化输出 — 使用 function_calling 方式强制约束输出 schema
        # (默认 json_mode 对通义千问不稳定, 模型可能返回数组或非法结构)
        self.structured_llm = llm.with_structured_output(
            LLMIntentResult, method="function_calling"
        )

        logger.info("[LLMRouter] 初始化完成 (结构化输出模式)")

    async def classify(self, query: str) -> IntentClassification:
        try:
            logger.info(f"[LLMRouter] L2 兜底分类: {query}")

            # LLM 直接返回 Pydantic 对象, 无需解析
            result: LLMIntentResult = await self.structured_llm.ainvoke(
                LLM_CLASSIFY_PROMPT.format(query=query)
            )

            logger.info(
                f"[LLMRouter] L2 结果: {result.intent} "
                f"(置信度: {result.confidence:.2f}, 理由: {result.reasoning})"
            )

            return IntentClassification(
                intent=result.intent,
                confidence=result.confidence,
                reasoning=result.reasoning,
                source="llm",
            )

        except Exception as e:
            logger.error(f"[LLMRouter] 分类失败: {e}", exc_info=True)
            return IntentClassification(
                intent="knowledge",
                confidence=0.5,
                reasoning=f"LLM 分类失败, 默认 knowledge: {str(e)}",
                source="llm",
            )


# =========================================================
# HybridRouter: 统一入口
# =========================================================

class HybridRouter:
    """
    混合意图路由器

    L1: SemanticRouter → 向量快速匹配 (~50ms)
    L2: LLMRouter → 结构化输出兜底 (~1-2s)
    """

    def __init__(self, llm=None, embedding_fn=None, threshold: float = 0.78):
        self.semantic_router = SemanticRouter(
            embedding_fn=embedding_fn,
            threshold=threshold,
        )
        self.llm_router = LLMRouter(llm=llm)

        logger.info(
            f"[HybridRouter] 初始化完成, "
            f"semantic threshold={threshold}"
        )

    async def classify(self, query: str) -> IntentClassification:
        """
        两层意图分类

        Returns:
            IntentClassification: 分类结果 (含 source 字段标明来源)
        """

        # L1: Semantic Router (快速路径)
        semantic_result = await self.semantic_router.classify(query)

        if semantic_result is not None:
            logger.info(
                f"[HybridRouter] L1 命中: {semantic_result.intent} "
                f"(score={semantic_result.confidence:.4f})"
            )
            return IntentClassification(
                intent=semantic_result.intent,
                confidence=semantic_result.confidence,
                reasoning=semantic_result.reasoning,
                source="semantic",
            )

        # L2: LLM Router (兜底)
        logger.info("[HybridRouter] L1 未命中, 进入 L2 LLM 兜底")
        return await self.llm_router.classify(query)
