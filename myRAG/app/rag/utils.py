"""
RAG 工具函数 (Pure Functions)

- format_docs: 将检索到的 Document 列表格式化为 LLM 可读的文本
- 包含 prompt injection 边界标记防护
"""

import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Prompt injection 边界标记
# 用不可见的结构化标记包裹文档内容，防止文档中内嵌的指令被 LLM 当作系统指令执行
_DOC_BOUNDARY_START = "<<<DOCUMENT_CONTENT>>>"
_DOC_BOUNDARY_END = "<<<END_DOCUMENT_CONTENT>>>"


def format_docs(documents: List[Document]) -> str:
    """
    将检索到的 Document 列表格式化为 LLM 可读的纯文本

    Args:
        documents: 检索到的 Document 列表

    Returns:
        格式化后的文本字符串
    """
    formatted_content = []
    for doc in documents:
        page_num = doc.metadata.get("page_num", "N/A")
        chunk_type = doc.metadata.get("chunk_type", "N/A")
        row_content = doc.page_content

        if chunk_type == "table":
            content = row_content
            section = f"第{page_num} 页: [表格类型]\n{_DOC_BOUNDARY_START}\n{content}\n{_DOC_BOUNDARY_END}"
        elif chunk_type == "image":
            content = row_content.replace("\n", " ")
            section = f"第{page_num} 页: [图片相关]\n{_DOC_BOUNDARY_START}\n{content}\n{_DOC_BOUNDARY_END}"
        else:
            content = row_content.replace("\n", " ")
            section = f"第{page_num} 页: [文本内容]\n{_DOC_BOUNDARY_START}\n{content}\n{_DOC_BOUNDARY_END}"

        formatted_content.append(section)

    return "\n\n".join(formatted_content)


# =========================================================
# Token 预算截断
# =========================================================

# --- Token 估算策略 ---
# 当前使用字符数估算: 1 个中文字符 ≈ 1.5~2 tokens
# 估算系数 CHARS_PER_TOKEN 取保守值, 宁可少留也不要溢出
#
# ⚠️ 后续选型可替换为:
#   - tiktoken (OpenAI tokenizer): 速度快、离线, 但和通义千问 tokenizer 不完全对齐
#   - llm.get_num_tokens(text): 精确匹配模型 tokenizer, 但需要 LLM 实例且略慢
#   - transformers AutoTokenizer: 最精确, 但依赖重
#
# 替换时只需修改 _estimate_tokens() 函数, 其余逻辑不变。
# =========================================================

# 每个 token 大约对应的字符数 (中文偏保守)
# 中文: ~1.5 chars/token, 英文: ~4 chars/token, 混合取 ~1.8
_CHARS_PER_TOKEN = 1.8


def _estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数

    当前实现: 字符数 / CHARS_PER_TOKEN (粗估)

    TODO: 后续可替换为精确 tokenizer:
        - tiktoken.encoding_for_model("gpt-4").encode(text)
        - llm.get_num_tokens(text)
    """
    return int(len(text) / _CHARS_PER_TOKEN)


def truncate_context(context: str, max_tokens: int = 3000) -> str:
    """
    按 token 预算截断 RAG 检索上下文

    策略:
    - 按文档段落 (双换行分隔) 逐段累加
    - 超出预算时截止, 保留完整段落 (不在段落中间截断)
    - 避免断句影响 LLM 理解

    Args:
        context: format_docs 返回的格式化文本
        max_tokens: token 预算上限 (默认 3000)

    Returns:
        截断后的上下文文本
    """
    if not context:
        return context

    total_tokens = _estimate_tokens(context)
    if total_tokens <= max_tokens:
        return context

    # 按文档段落分割 (format_docs 用 \n\n 分隔各文档)
    sections = context.split("\n\n")
    kept_sections = []
    used_tokens = 0

    for section in sections:
        section_tokens = _estimate_tokens(section)
        if used_tokens + section_tokens > max_tokens:
            break
        kept_sections.append(section)
        used_tokens += section_tokens

    # 至少保留一个段落 (即使超预算)
    if not kept_sections and sections:
        kept_sections.append(sections[0])
        used_tokens = _estimate_tokens(sections[0])

    truncated = "\n\n".join(kept_sections)

    dropped = len(sections) - len(kept_sections)
    if dropped > 0:
        logger.warning(
            f"[TokenBudget] 上下文截断: 保留 {len(kept_sections)}/{len(sections)} 段, "
            f"~{used_tokens}/{total_tokens} tokens (预算 {max_tokens})"
        )

    return truncated
