"""
检索器管线测试

覆盖:
- get_relevant_documents 完整流程 (mock 向量 + BM25)
- parent_id 去重
- top_k 截断
- BM25 为 None 时的降级
- reload_bm25 原子性
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from langchain_core.documents import Document

from conftest import make_doc


def _build_retriever(top_k=2, recall_k=5):
    """构建测试用 retriever, mock 掉所有外部依赖"""
    with patch("app.rag.retriever.get_tencent_embeddings") as mock_emb, \
         patch("app.rag.retriever.Chroma") as mock_chroma, \
         patch("app.rag.retriever.SQLAlchemyDocStore") as mock_sql, \
         patch("app.rag.retriever.BM25Retriever") as mock_bm25_cls:

        # Mock Chroma
        mock_db = MagicMock()
        mock_db.get.return_value = {"documents": [], "metadatas": []}
        mock_chroma.return_value = mock_db

        # Mock vector retriever
        mock_vec_retriever = MagicMock()
        mock_db.as_retriever.return_value = mock_vec_retriever

        # Mock SQL store
        mock_store = MagicMock()
        mock_sql.return_value = mock_store

        from app.rag.retriever import AdvancedRetriever
        retriever = AdvancedRetriever(top_k=top_k, recall_k=recall_k, db_url="mock://db")

        # 暴露 mock 引用以便测试中设置行为
        retriever._mock_vec = mock_vec_retriever
        retriever._mock_sql = mock_store

        return retriever


class TestRetrieverPipeline:
    """get_relevant_documents 完整管线测试"""

    def test_full_pipeline(self):
        """完整管线: 向量检索 → RRF → parent_id 去重 → MySQL 回溯"""
        retriever = _build_retriever(top_k=2)

        # 设置向量检索返回
        child1 = make_doc("子文档1", parent_id="p001", file_hash="f1")
        child2 = make_doc("子文档2", parent_id="p002", file_hash="f2")
        retriever._mock_vec.invoke.return_value = [child1, child2]

        # BM25 禁用 (设为 None)
        retriever.bm25_retriever = None

        # 设置 MySQL 回溯返回
        parent1 = make_doc("父文档1", parent_id="p001")
        parent2 = make_doc("父文档2", parent_id="p002")
        retriever._mock_sql.get_parent_by_id.side_effect = lambda pid: {
            "p001": parent1,
            "p002": parent2,
        }.get(pid)

        result = retriever.get_relevant_documents("测试查询")

        assert len(result) == 2
        assert result[0].page_content == "父文档1"
        assert result[1].page_content == "父文档2"

    def test_parent_id_dedup(self):
        """相同 parent_id 的子文档应去重, 只回溯一次"""
        retriever = _build_retriever(top_k=5)

        # 3 个子文档, 但只有 2 个不同的 parent_id
        child1 = make_doc("子1", parent_id="p001", file_hash="f1")
        child2 = make_doc("子2", parent_id="p001", file_hash="f2")  # 同一个 parent
        child3 = make_doc("子3", parent_id="p002", file_hash="f3")
        retriever._mock_vec.invoke.return_value = [child1, child2, child3]
        retriever.bm25_retriever = None

        parent1 = make_doc("父1")
        parent2 = make_doc("父2")
        retriever._mock_sql.get_parent_by_id.side_effect = lambda pid: {
            "p001": parent1,
            "p002": parent2,
        }.get(pid)

        result = retriever.get_relevant_documents("去重测试")

        # 应该只有 2 个父文档
        assert len(result) == 2

    def test_top_k_truncation(self):
        """超过 top_k 的结果应被截断"""
        retriever = _build_retriever(top_k=1)  # 只要 1 个

        child1 = make_doc("子1", parent_id="p001", file_hash="f1")
        child2 = make_doc("子2", parent_id="p002", file_hash="f2")
        retriever._mock_vec.invoke.return_value = [child1, child2]
        retriever.bm25_retriever = None

        parent1 = make_doc("父1")
        retriever._mock_sql.get_parent_by_id.side_effect = lambda pid: {
            "p001": parent1,
        }.get(pid)

        result = retriever.get_relevant_documents("截断测试")

        # top_k=1, 应只返回 1 个
        assert len(result) == 1

    def test_no_parent_id_returns_empty(self):
        """没有 parent_id 的文档应被跳过, 最终返回空"""
        retriever = _build_retriever(top_k=5)

        doc_no_parent = Document(
            page_content="无parent",
            metadata={"file_hash": "x1", "page": 1, "type": "text"}
            # 没有 parent_id
        )
        retriever._mock_vec.invoke.return_value = [doc_no_parent]
        retriever.bm25_retriever = None

        result = retriever.get_relevant_documents("无parent测试")
        assert result == []

    def test_bm25_none_graceful_degradation(self):
        """BM25 为 None 时应优雅降级, 只使用向量检索"""
        retriever = _build_retriever(top_k=2)
        retriever.bm25_retriever = None  # 明确设为 None

        child1 = make_doc("仅向量", parent_id="p001", file_hash="f1")
        retriever._mock_vec.invoke.return_value = [child1]

        parent1 = make_doc("父文档")
        retriever._mock_sql.get_parent_by_id.return_value = parent1

        # 不应抛异常
        result = retriever.get_relevant_documents("降级测试")
        assert len(result) == 1

    def test_reload_bm25_atomic(self):
        """reload_bm25 应先构建新索引, 再替换引用"""
        retriever = _build_retriever()
        old_bm25 = retriever.bm25_retriever

        # Mock _build_bm25 返回新对象
        new_bm25 = MagicMock()
        with patch.object(retriever, "_build_bm25", return_value=new_bm25):
            retriever.reload_bm25()

        # 应该已替换
        assert retriever.bm25_retriever is new_bm25
        assert retriever.bm25_retriever is not old_bm25
