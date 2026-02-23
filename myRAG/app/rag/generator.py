"""
RAG Generator (已废弃)

format_docs 已迁移到 app.rag.utils 模块。
此文件仅保留向后兼容的 re-export。

如果你看到从这里 import format_docs, 请改为:
    from app.rag.utils import format_docs
"""

# 向后兼容: 防止旧代码 import 报错
from app.rag.utils import format_docs  # noqa: F401
