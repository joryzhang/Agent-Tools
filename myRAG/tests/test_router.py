"""
意图路由器测试

覆盖:
- HybridRouter: L1 命中 → 直接返回
- HybridRouter: L1 未命中 → L2 LLM 兜底
- LLMRouter: 异常 fallback
- IntentClassification 数据模型
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.router import (
    HybridRouter,
    LLMRouter,
    IntentClassification,
    LLMIntentResult,
)
from app.agent.semantic_router import SemanticClassification


class TestIntentClassification:
    """IntentClassification 数据模型测试"""

    def test_valid_intent(self):
        """合法意图类型应能正常创建"""
        for intent in ["user_info", "knowledge", "mixed", "normal"]:
            ic = IntentClassification(
                intent=intent,
                confidence=0.9,
                reasoning="测试",
                source="test",
            )
            assert ic.intent == intent

    def test_confidence_bounds(self):
        """置信度应在 0-1 范围内"""
        ic = IntentClassification(
            intent="knowledge",
            confidence=0.0,
            reasoning="最低",
            source="test",
        )
        assert ic.confidence == 0.0

        ic2 = IntentClassification(
            intent="knowledge",
            confidence=1.0,
            reasoning="最高",
            source="test",
        )
        assert ic2.confidence == 1.0


class TestHybridRouter:
    """HybridRouter L1 + L2 集成测试"""

    @pytest.mark.asyncio
    async def test_l1_hit(self):
        """L1 SemanticRouter 命中时应直接返回, 不调用 L2"""
        router = HybridRouter.__new__(HybridRouter)

        # Mock SemanticRouter: 返回命中结果
        router.semantic_router = AsyncMock()
        router.semantic_router.classify.return_value = SemanticClassification(
            intent="knowledge",
            confidence=0.92,
            reasoning="匹配到知识库模板",
        )

        # Mock LLMRouter (不应被调用)
        router.llm_router = AsyncMock()

        result = await router.classify("VIP 有什么特权?")

        assert result.intent == "knowledge"
        assert result.source == "semantic"
        assert result.confidence == 0.92
        router.llm_router.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_l1_miss_l2_fallback(self):
        """L1 未命中时应 fallback 到 L2 LLM"""
        router = HybridRouter.__new__(HybridRouter)

        # Mock SemanticRouter: 返回 None (未命中)
        router.semantic_router = AsyncMock()
        router.semantic_router.classify.return_value = None

        # Mock LLMRouter: 返回结果
        router.llm_router = AsyncMock()
        router.llm_router.classify.return_value = IntentClassification(
            intent="user_info",
            confidence=0.85,
            reasoning="用户问余额",
            source="llm",
        )

        result = await router.classify("我的余额是多少?")

        assert result.intent == "user_info"
        assert result.source == "llm"
        router.llm_router.classify.assert_called_once()


class TestLLMRouter:
    """LLMRouter 异常处理测试"""

    @pytest.mark.asyncio
    async def test_llm_exception_fallback(self):
        """LLM 调用失败时应 fallback 到 knowledge"""
        router = LLMRouter.__new__(LLMRouter)

        # Mock structured LLM: 抛异常
        router.structured_llm = AsyncMock()
        router.structured_llm.ainvoke.side_effect = RuntimeError("API 超时")

        result = await router.classify("测试问题")

        # 应 fallback 到 knowledge, 置信度 0.5
        assert result.intent == "knowledge"
        assert result.confidence == 0.5
        assert "失败" in result.reasoning
        assert result.source == "llm"
