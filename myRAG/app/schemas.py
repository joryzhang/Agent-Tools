from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# ==========================================
# 基础组件 Schema
# ==========================================

class SourceInfo(BaseModel):
    """
    引用来源的结构定义
    告诉前端：这段话是参考哪里的
    """
    file_name: str = Field(..., description="文件名")
    page: int = Field(..., description="页码")
    score: float = Field(0.0, description="检索匹配度/RRF得分")
    content_snippet: str = Field(..., description="原文片段(前100字)", max_length=200)
    file_hash: str = Field(..., description="文件哈希ID，用于前端跳转")


# ==========================================
# 请求 (Request) Schema
# ==========================================


class ChatRequest(BaseModel):
    """
    前端发来的聊天请求格式
    """
    query: str = Field(
        ...,
        description="用户的提问",
        min_length=1,
        max_length=2000,
        examples=["违约金比例是多少？"]
    )

    # 对话历史 (暂留接口，后续支持多轮对话)
    history: List[Dict[str, str]] = Field(
        default=[],
        description="历史对话上下文，格式: [{'role': 'user', 'content': '...'}, ...]"
    )

    # 高级参数 (允许前端微调)
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索切片数量"
    )

    # LLM 温度 (请求级参数, RAG 路径会被后端强制覆盖为 0)
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM 生成温度: 0=精确, 1=创造性。RAG 路径强制为 0"
    )

    enable_streaming: bool = Field(
        default=True,
        description="是否开启流式响应"
    )


# ==========================================
# 响应 (Response) Schema
# ==========================================

class ChatResponse(BaseModel):
    """
    非流式接口的完整响应格式
    """
    answer: str = Field(..., description="RAG 生成的最终回答")
    sources: List[SourceInfo] = Field(default=[], description="参考的文档列表")
    metadata: Dict[str, Any] = Field(default={}, description="额外的元数据(耗时、Token消耗等)")
