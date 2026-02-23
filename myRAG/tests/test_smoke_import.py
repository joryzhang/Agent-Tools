"""
冒烟测试 — 确保所有核心模块可以被正常 import

防止: 缺失依赖、循环 import、语法错误等低级问题。
"""
import pytest


@pytest.mark.parametrize("module", [
    "app.config",
    "app.server",
    "app.api.vector_admin",
    "app.api.rag_config_api",
    "app.agent.executor",
    "app.agent.router",
    "app.agent.semantic_router",
    "app.tools.rag_tool",
    "app.tools.user_info_tool",
    "app.rag.utils",
    "app.rag.retriever",
    "app.rag.generator",
    "app.ingest.service",
    "app.database.orchestrator",
    "app.database.storage",
    "app.database.VectorStoreManager",
    "app.middleware.auth",
])
def test_module_importable(module):
    """每个核心模块都应能无错误导入"""
    __import__(module)
