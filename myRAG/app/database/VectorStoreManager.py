import os
import logging
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from models import get_tencent_embeddings

from app.config import DATA_DIR

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, collection_name: str = "rag_collection"):
        """
        初始化向量数据库管理器
        注意：这里不再初始化 TextSplitter，因为切分工作已移交给 Processor
        """
        # 1. 设置持久化路径
        self.persist_directory = os.path.join(DATA_DIR, "chroma_db")
        self.collection_name = collection_name

        # 2. 初始化 Embedding 模型
        self.embedding_fn = get_tencent_embeddings()

        # 3. 初始化 ChromaDB
        self.vector_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_fn,
            persist_directory=self.persist_directory
        )

    def file_exists(self, file_hash: str) -> bool:
        """
        检查文件是否已存在 (幂等性检查)
        """
        if not file_hash:
            return False

        results = self.vector_db.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results["ids"]) > 0  # 是由 ChromaDB自动生成的，我们改了add_document方法修改了ids，确保我们在多次运行的时候不会重复传入相同数据

    def add_documents(self, documents: List[Document]) -> bool:
        """
        核心方法：将子文档 (Child Documents) 存入向量库

        Args:
            documents: 这里接收的应该是 Processor 切分好的 child_docs
        """
        if not documents:
            return False

        # 1. 检查文件是否已存在 (基于第一个文档的 hash)
        # 假设同一批文档来自同一个文件
        file_hash = documents[0].metadata.get("file_hash")

        if file_hash and self.file_exists(file_hash):
            logger.info(f"[VectorStore] 文件 ({file_hash}) 已存在，跳过向量化")
            return False

        logger.info(f"[VectorStore] 接收到 {len(documents)} 个子文档，准备入库...")

        # 提取我们在 Processor 里生成的 ID
        # 注意：前提是你在 processor 里给 doc 赋值了 id 属性，或者存在 metadata 里
        ids = [doc.id for doc in documents]  # 一个id列表

        # 2. 批量写入
        # 直接存入，因为 Processor 已经把它们切成了适合的大小 (e.g. 400 tokens)
        batch_size = 100
        total_chunks = len(documents)

        for i in range(0, total_chunks, batch_size):
            batch_docs = documents[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]
            self.vector_db.add_documents(documents=batch_docs, ids=batch_ids)
            logger.debug(f"  - 向量写入批次 {i // batch_size + 1}/{(total_chunks // batch_size) + 1}")

        logger.info(f"[VectorStore] 入库成功！共存储 {total_chunks} 个向量块")
        return True

    def search(self, query: str, k: int = 3):
        """
        检索功能：返回相关的子文档
        """
        logger.info(f"[VectorStore] 检索: {query}")
        # 增加 filter 示例：如果你只想搜特定文件，可以加 filter={"file_hash": "xxx"}
        results = self.vector_db.similarity_search_with_score(query, k=k)

        for doc, score in results:
            # 打印 parent_id，证明我们能反向找到父文档
            p_id = doc.metadata.get('parent_id', 'N/A')
            logger.debug(f"  [Score: {score:.4f}] ParentID: {p_id} | Page: {doc.metadata.get('page')}")
            logger.debug(f"  {doc.page_content[:100].replace(chr(10), ' ')}...")

        return results

    def delete_file(self, file_hash: str):
        """
        [新增] 删除指定文件的所有向量 (用于重新解析时清理旧数据)
        """
        self.vector_db._collection.delete(where={"file_hash": file_hash})
        logger.info(f"[VectorStore] 已删除文件 {file_hash} 的向量索引")

    def clear_db(self):
        """物理清空数据库"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logger.warning("[VectorStore] 数据库已物理删除")
