"""
RAGTool 测试

覆盖:
- retriever=None 时的安全返回
- 无检索结果时的提示
- 正常检索 + 格式化
"""
import pytest
from unittest.mock import MagicMock
from app.tools.rag_tool import EnterpriseRAGTool

from conftest import make_doc


class TestRAGTool:
    """EnterpriseRAGTool 核心测试"""

    def test_retriever_none(self):
        """retriever 未初始化时应返回友好提示, 不崩溃"""
        tool = EnterpriseRAGTool(retriever=None)
        result = tool._run(query="测试")
        assert "未初始化" in result

    def test_no_results(self):
        """检索无结果时应返回提示"""
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.return_value = []

        tool = EnterpriseRAGTool(retriever=mock_retriever)
        result = tool._run(query="不存在的内容")
        assert "未在知识库中找到" in result

    def test_normal_retrieval(self):
        """正常检索应返回格式化的文档内容"""
        mock_retriever = MagicMock()
        doc = make_doc("这是知识库中的内容", chunk_type="text", page_num=1)
        mock_retriever.get_relevant_documents.return_value = [doc]

        tool = EnterpriseRAGTool(retriever=mock_retriever)
        result = tool._run(query="测试查询")

        assert "这是知识库中的内容" in result
        assert "[文本内容]" in result
        mock_retriever.get_relevant_documents.assert_called_once_with("测试查询")

    def test_retriever_exception(self):
        """检索器抛异常时应捕获并返回错误信息"""
        mock_retriever = MagicMock()
        mock_retriever.get_relevant_documents.side_effect = RuntimeError("连接失败")

        tool = EnterpriseRAGTool(retriever=mock_retriever)
        result = tool._run(query="异常测试")

        assert "出错" in result
        assert "连接失败" in result
