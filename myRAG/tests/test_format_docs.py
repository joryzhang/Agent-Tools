"""
format_docs + truncate_context 纯函数测试

覆盖:
- 文本/表格/图片类型的格式化
- 空列表处理
- prompt injection 边界标记验证
- Token 预算截断
"""
import pytest
from langchain_core.documents import Document
from app.rag.utils import format_docs, _DOC_BOUNDARY_START, _DOC_BOUNDARY_END

# 复用 conftest 中的 make_doc
from conftest import make_doc


class TestFormatDocs:
    """format_docs 核心测试"""

    def test_empty_list(self):
        """空文档列表应返回空字符串"""
        assert format_docs([]) == ""

    def test_text_document(self, sample_text_doc):
        """文本类型应显示 [文本内容] 标签"""
        result = format_docs([sample_text_doc])
        assert "[文本内容]" in result
        assert "第3 页" in result
        assert "这是一段测试文本内容" in result

    def test_table_document(self, sample_table_doc):
        """表格类型应显示 [表格类型] 标签"""
        result = format_docs([sample_table_doc])
        assert "[表格类型]" in result
        assert "第5 页" in result

    def test_image_document(self, sample_image_doc):
        """图片类型应显示 [图片相关] 标签, 且换行被替换"""
        result = format_docs([sample_image_doc])
        assert "[图片相关]" in result
        assert "第7 页" in result
        # 图片内容的换行应被替换为空格
        assert "图片描述: 某个架构图 第二行" in result

    def test_boundary_markers_present(self, sample_text_doc):
        """每个文档内容都应被边界标记包裹 (prompt injection 防护)"""
        result = format_docs([sample_text_doc])
        assert _DOC_BOUNDARY_START in result
        assert _DOC_BOUNDARY_END in result
        # 边界标记应成对出现
        assert result.count(_DOC_BOUNDARY_START) == 1
        assert result.count(_DOC_BOUNDARY_END) == 1

    def test_multiple_docs_boundary_count(self, mixed_docs):
        """多文档时, 每个文档都应有独立的边界标记"""
        result = format_docs(mixed_docs)
        assert result.count(_DOC_BOUNDARY_START) == 3
        assert result.count(_DOC_BOUNDARY_END) == 3

    def test_docs_separated_by_double_newline(self, mixed_docs):
        """多文档之间应用双换行分隔"""
        result = format_docs(mixed_docs)
        sections = result.split("\n\n")
        # 3 个文档, 每个文档内部有多行, 总 section 数 > 3
        assert len(sections) >= 3

    def test_text_newlines_replaced(self):
        """文本类型的换行应被替换为空格"""
        doc = make_doc("第一行\n第二行\n第三行", chunk_type="text")
        result = format_docs([doc])
        assert "第一行 第二行 第三行" in result

    def test_table_preserves_newlines(self):
        """表格类型应保留换行 (Markdown 表格需要)"""
        doc = make_doc("| A |\n|---|\n| 1 |", chunk_type="table")
        result = format_docs([doc])
        assert "| A |\n|---|\n| 1 |" in result


# =========================================================
# truncate_context 测试
# =========================================================
from app.rag.utils import truncate_context, _estimate_tokens


class TestEstimateTokens:
    """_estimate_tokens 估算测试"""

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_chinese_text(self):
        """中文文本估算应大于 0"""
        tokens = _estimate_tokens("这是一段测试文本")
        assert tokens > 0

    def test_longer_text_more_tokens(self):
        """更长的文本应有更多 token"""
        short = _estimate_tokens("短文本")
        long = _estimate_tokens("这是一段非常非常非常长的测试文本内容")
        assert long > short


class TestTruncateContext:
    """truncate_context Token 预算截断测试"""

    def test_empty_context(self):
        """空字符串应原样返回"""
        assert truncate_context("", max_tokens=100) == ""

    def test_under_budget_passthrough(self):
        """未超预算时应原样返回"""
        short_text = "第1页: [文本内容]\n短文本"
        result = truncate_context(short_text, max_tokens=10000)
        assert result == short_text

    def test_over_budget_truncation(self):
        """超预算时应截断, 且结果比原文短"""
        # 构造一个很长的上下文 (多个段落)
        sections = [f"文档{i}: " + "A" * 500 for i in range(10)]
        long_context = "\n\n".join(sections)

        result = truncate_context(long_context, max_tokens=200)

        # 结果应比原文短
        assert len(result) < len(long_context)
        # 结果中不应有被截断的段落 (保留完整段落)
        for section in result.split("\n\n"):
            assert section in sections

    def test_at_least_one_section_kept(self):
        """即使单个段落就超预算, 也应保留至少一个"""
        huge_section = "A" * 10000
        result = truncate_context(huge_section, max_tokens=1)

        # 应保留整个段落
        assert result == huge_section

    def test_preserves_section_order(self):
        """截断后的段落顺序应与原文一致"""
        sections = ["第一段内容AAA", "第二段内容BBB", "第三段内容CCC"]
        context = "\n\n".join(sections)

        result = truncate_context(context, max_tokens=20)

        result_sections = result.split("\n\n")
        # 保留的段落应从前往后, 不应跳段
        for i, section in enumerate(result_sections):
            assert section == sections[i]
