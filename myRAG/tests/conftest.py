"""
共享 Fixtures — 为所有测试提供 mock Document 和工具函数
"""
import pytest
from langchain_core.documents import Document


# =========================================================
# 文档工厂
# =========================================================

def make_doc(content: str = "测试内容", **meta_overrides) -> Document:
    """
    快速创建 Document 对象

    默认 metadata 包含所有常用字段, 可通过 meta_overrides 覆盖。
    """
    metadata = {
        "page_num": 1,
        "page": 1,
        "chunk_type": "text",
        "file_hash": "abc123",
        "parent_id": "p001",
        "type": "text",
        "media_path": "",
    }
    metadata.update(meta_overrides)
    return Document(page_content=content, metadata=metadata)


@pytest.fixture
def sample_text_doc():
    """一个文本类型的 Document"""
    return make_doc("这是一段测试文本内容", chunk_type="text", page_num=3)


@pytest.fixture
def sample_table_doc():
    """一个表格类型的 Document"""
    return make_doc("| 列A | 列B |\n|---|---|\n| 1 | 2 |", chunk_type="table", page_num=5)


@pytest.fixture
def sample_image_doc():
    """一个图片类型的 Document"""
    return make_doc("图片描述: 某个架构图\n第二行", chunk_type="image", page_num=7)


@pytest.fixture
def mixed_docs(sample_text_doc, sample_table_doc, sample_image_doc):
    """混合类型文档列表"""
    return [sample_text_doc, sample_table_doc, sample_image_doc]
