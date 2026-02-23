"""
EnterpriseRAGTool - 企业级 RAG 检索工具 (纯检索版)

职责: 只负责检索 + 格式化文档, 不调用 LLM 生成。
LLM 生成由 Agent 节点统一负责。

设计原则:
- 不在 Tool 内部创建依赖 (IoC)
- retriever 由外部注入
- 不依赖 generator (已移除)
"""

from typing import Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import logging
from app.rag.utils import format_docs

logger = logging.getLogger(__name__)


class RAGQueryInput(BaseModel):
    """RAG 查询输入"""
    query: str = Field(description="用户的问题")


class EnterpriseRAGTool(BaseTool):
    """
    企业级 RAG 检索工具 (纯检索)

    只负责从知识库中检索相关文档并返回格式化的原文内容。
    不做 LLM 生成, LLM 生成由 Agent 节点统一负责。

    使用场景:
    - agent 节点判断需要额外知识库信息时主动调用
    - mixed 意图下, agent 可同时调用此工具和 user_info 工具
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "search_knowledge_base"
    description: str = (
        "从企业知识库中检索相关文档。"
        "当用户询问业务规则、公司政策、产品说明、技术文档等问题时使用。"
        "输入 query, 返回检索到的原始文档内容。"
    )

    # RAG 组件 - 由外部注入
    retriever: Optional[Any] = None

    def _run(self, query: str) -> str:
        """
        执行知识库检索 (纯检索, 不生成)

        Returns:
            格式化的文档原文内容
        """
        try:
            logger.info(f"[RAGTool] 检索知识库: {query}")

            if self.retriever is None:
                return "RAG 检索器未初始化, 请联系管理员"

            # 1. 检索文档
            docs = self.retriever.get_relevant_documents(query)

            if not docs:
                return "未在知识库中找到相关文档。"

            # 2. 格式化文档内容
            result = format_docs(docs)

            logger.info(f"[RAGTool] 检索成功, 返回 {len(docs)} 个文档, 内容长度: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"[RAGTool] 检索失败: {e}", exc_info=True)
            return f"检索知识库时出错: {str(e)}"
