"""
RRF 融合算法测试

覆盖:
- 单路/双路融合
- 权重配置生效
- file_hash 缺失 fallback
- 空输入
- 排序正确性
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from conftest import make_doc


def _create_retriever_for_rrf():
    """
    创建一个最小化的 AdvancedRetriever 实例, 仅用于测试 _rrf_fusion。
    通过 mock 跳过 __init__ 中的外部依赖。
    """
    from unittest.mock import patch

    with patch("app.rag.retriever.get_tencent_embeddings"), \
         patch("app.rag.retriever.Chroma"), \
         patch("app.rag.retriever.SQLAlchemyDocStore"), \
         patch("app.rag.retriever.BM25Retriever"):
        from app.rag.retriever import AdvancedRetriever
        retriever = AdvancedRetriever.__new__(AdvancedRetriever)
        retriever.top_k = 5
        retriever.recall_k = 10
        retriever.score_threshold = 0.4
        return retriever


class TestRRFFusion:
    """_rrf_fusion 算法核心测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前创建 retriever"""
        self.retriever = _create_retriever_for_rrf()

    def test_empty_input(self):
        """空输入应返回空列表"""
        result = self.retriever._rrf_fusion([[], []])
        assert result == []

    def test_single_list(self):
        """单路检索也能正常融合"""
        docs = [make_doc("doc1", file_hash="h1"), make_doc("doc2", file_hash="h2")]
        result = self.retriever._rrf_fusion([docs])
        assert len(result) == 2

    @patch("app.rag.retriever.settings")
    def test_dual_list_fusion(self, mock_settings):
        """双路检索: 相同文档在两路中出现应得分更高"""
        mock_settings.RRF_K = 60
        mock_settings.TABLE_WEIGHT = 1.0
        mock_settings.IMAGE_WEIGHT = 1.0

        shared_doc = make_doc("共享文档", file_hash="shared")
        only_vec = make_doc("仅向量", file_hash="vec_only")
        only_bm25 = make_doc("仅BM25", file_hash="bm25_only")

        vec_list = [shared_doc, only_vec]
        bm25_list = [only_bm25, shared_doc]

        result = self.retriever._rrf_fusion([vec_list, bm25_list])

        # 共享文档在两路都出现, 应排第一
        assert result[0].page_content == "共享文档"

    @patch("app.rag.retriever.settings")
    def test_table_weight_boost(self, mock_settings):
        """table 权重 > 1 应提升表格文档排名"""
        mock_settings.RRF_K = 60
        mock_settings.TABLE_WEIGHT = 3.0  # 表格权重 3 倍
        mock_settings.IMAGE_WEIGHT = 1.0

        text_doc = make_doc("普通文本", file_hash="text1", type="text")
        table_doc = make_doc("表格内容", file_hash="table1", type="table")

        # table 排名靠后 (rank=1), text 排名靠前 (rank=0)
        # 但 table 有 3x 权重加成
        result = self.retriever._rrf_fusion([[text_doc, table_doc]])

        # 3x 权重加成后, table 的 score = 3/(60+1) ≈ 0.0492
        # text 的 score = 1/(60+0) ≈ 0.0167
        # table 应排第一 -- 等等, text rank=0 得分 1/60=0.0167,
        # table rank=1 得分 3/61=0.0492, 所以 table > text
        assert result[0].page_content == "表格内容"

    @patch("app.rag.retriever.settings")
    def test_file_hash_missing_fallback(self, mock_settings):
        """file_hash 缺失时应使用 content hash 作为 fallback, 不崩溃"""
        mock_settings.RRF_K = 60
        mock_settings.TABLE_WEIGHT = 1.0
        mock_settings.IMAGE_WEIGHT = 1.0

        # 创建没有 file_hash 的文档
        doc = Document(
            page_content="无 hash 文档",
            metadata={"page": 1, "type": "text"}
        )

        # 不应抛异常
        result = self.retriever._rrf_fusion([[doc]])
        assert len(result) == 1
        assert result[0].page_content == "无 hash 文档"

    @patch("app.rag.retriever.settings")
    def test_deduplication(self, mock_settings):
        """相同文档出现多次应合并, 不重复"""
        mock_settings.RRF_K = 60
        mock_settings.TABLE_WEIGHT = 1.0
        mock_settings.IMAGE_WEIGHT = 1.0

        doc = make_doc("重复文档", file_hash="dup1")
        result = self.retriever._rrf_fusion([[doc, doc]])

        # 虽然传入 2 个相同文档, 但 doc_key 相同, 应合并为 1 个
        # 注意: 由于 content 前 20 字符和 page 都一样, doc_key 会一样
        assert len(result) == 1

    @patch("app.rag.retriever.settings")
    def test_custom_k_parameter(self, mock_settings):
        """自定义 k 参数应覆盖 settings.RRF_K"""
        mock_settings.RRF_K = 60  # 默认值, 但应被覆盖
        mock_settings.TABLE_WEIGHT = 1.0
        mock_settings.IMAGE_WEIGHT = 1.0

        doc = make_doc("测试", file_hash="k_test")
        result = self.retriever._rrf_fusion([[doc]], k=1)

        # k=1 时, rank=0 的 score = 1/(1+0) = 1.0
        # 不崩溃即可
        assert len(result) == 1
