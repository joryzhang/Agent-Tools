import concurrent.futures
import hashlib
import logging
import time
from typing import List, Dict, Any

import jieba
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.config import DATA_DIR, settings
from app.database.storage import SQLAlchemyDocStore  # MySQL 连接
from models import get_tencent_embeddings

logger = logging.getLogger(__name__)


# --- 1. 定义中文分词预处理函数 ---
def chinese_tokenizer(text: str) -> List[str]:
    """
    使用 jieba 进行中文分词。
    BM25 需要把句子变成词列表 ["我", "爱", "北京"] 才能计算词频。
    """
    return list(jieba.cut_for_search(text))


class AdvancedRetriever:
    def __init__(self, top_k: int = 5, recall_k: int = 10, score_threshold: float = 0.4, db_url: str = None):
        """
        初始化硬核检索器
        Args:
            top_k: 最终返回的父文档数量 (送给 LLM 的上下文数)
            recall_k: 每路检索器的子切片召回数量 (撒网阶段, 应远大于 top_k)
            score_threshold: 相似度阈值
        """
        self.top_k = top_k
        self.recall_k = recall_k
        self.score_threshold = score_threshold
        # 1. 初始化向量数据库 (只读模式)
        self.embedding_fn = get_tencent_embeddings()
        self.vector_db = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_fn,
            persist_directory=f"{DATA_DIR}/chroma_db"
        )
        self.vector_retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.recall_k}  # 用 recall_k 扩大召回窗口
        )
        # 初始化bm25,加载chroma中的数据 // todo 生产环境需要换成elastic search
        self.bm25_retriever = self._build_bm25()
        # 2. 初始化关系型数据库 (用于查询原始分块)
        self.sql_store = SQLAlchemyDocStore(db_uri=db_url)

    def _rrf_fusion(self, list_of_list_docs: List[List[Document]], k=None) -> List[Document]:
        """
        实现倒数排名融合 (RRF) 算法
        :param list_of_list_docs: 检索结果列表 [ [doc_a, doc_b], [doc_a, doc_c] ]
        :param k: 平滑常数, 默认读取 settings.RRF_K (60)
                   k 越大 → 排名差异越平滑; k 越小 → 头部排名权重越大 ("赢者通吃")
        """
        if k is None:
            k = settings.RRF_K
        rrf_map = {}

        for doc_list in list_of_list_docs:
            for rank, doc in enumerate(doc_list):
                # 生成唯一键: file_hash + page + content_hash
                content_preview = doc.page_content[:20]
                stable_content_hash = hashlib.md5(content_preview.encode()).hexdigest()
                page_num = str(doc.metadata.get("page", "0"))
                file_hash = doc.metadata.get("file_hash", stable_content_hash)  # fallback 防 KeyError
                doc_key = file_hash + page_num + stable_content_hash

                if doc_key not in rrf_map:
                    rrf_map[doc_key] = {"doc": doc, "score": 0.0}
                # 基础分数
                score = 1.0 / (k + rank)

                # 可定制加权 (从 settings 动态读取)
                if doc.metadata.get("type") == "table":
                    score *= settings.TABLE_WEIGHT

                if doc.metadata.get("type") == "image":
                    score *= settings.IMAGE_WEIGHT

                rrf_map[doc_key]["score"] += score

        # 按分数倒序排序
        sorted_docs = sorted(rrf_map.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def reload_bm25(self):
        """
        强制重新构建 BM25 索引 (当有新文件入库时调用)
        使用"先构建后替换"策略，避免并发查询读到 None
        """
        logger.info("[BM25] 检测到数据变更，正在热更新索引...")
        new_bm25 = self._build_bm25()
        self.bm25_retriever = new_bm25  # 原子赋值替换
        logger.info("[BM25] 热更新完成")

    def _build_bm25(self):
        """
        从 Chroma 中提取所有文本构建内存级 BM25 索引。
        注意：生产环境数据量大时，这一步应该由 Elasticsearch 替代。
        """
        try:
            logger.info("[Hybrid] 正在从 Chroma 加载数据以构建 BM25 索引...")
            # 1. 获取所有数据 用chroma的get方法
            all_data = self.vector_db.get()
            texts = all_data['documents']
            metadatas = all_data['metadatas']
            if not texts:
                logger.warning("[Hybrid] 向量库为空，跳过 BM25 构建")
                return None

            # 2. 构造 Document 对象列表
            docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metadatas or [{}] * len(texts))
            ]

            bm25 = BM25Retriever.from_documents(
                documents=docs,
                preprocess=chinese_tokenizer
            )
            bm25.k = self.recall_k  # 用 recall_k 扩大召回窗口

            return bm25
        except Exception as e:
            logger.error(f"[Hybrid] BM25 构建出错: {e}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        :param query:  输入的问题
        :return: 包装后的Document, 方便LLM直接调用
        """
        start_time = time.time()
        logger.info(f"[Retriever] 正在检索: {query}")

        # 1. 第一阶段：并发执行两路检索 (Parallel Execution)
        bm25_docs = []
        embedding_docs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_vec = executor.submit(self.vector_retriever.invoke, query)
            future_bm25 = None
            if self.bm25_retriever:
                future_bm25 = executor.submit(self.bm25_retriever.invoke, query)

            try:
                embedding_docs = future_vec.result()
                if future_bm25:
                    bm25_docs = future_bm25.result()
            except Exception as e:
                logger.error(f"[Retriever] 并发检索出错: {e}")

        fused_child_docs = self._rrf_fusion([bm25_docs, embedding_docs])
        logger.info(f"[Hybrid] 召回 {len(fused_child_docs)} 个混合切片线索")

        # 2. 第二阶段：ID 提取与去重 (Deduplication)
        parent_ids = []
        seen_ids = set()

        for doc in fused_child_docs:
            p_id = doc.metadata.get("parent_id")
            if p_id and p_id not in seen_ids:
                parent_ids.append(p_id)
                seen_ids.add(p_id)
            # 限制最终送给 LLM 的文档数，防止 Token 爆炸
            if len(parent_ids) >= self.top_k:
                break

        if not parent_ids:
            return []

        # 3. 第三阶段：MySQL 回溯 (查大白话)
        logger.info(f"[Hybrid] 正在回溯 {len(parent_ids)} 个父文档...")
        final_docs = []
        for p_id in parent_ids:
            parent_doc = self.sql_store.get_parent_by_id(p_id)
            if parent_doc:
                final_docs.append(parent_doc)

        logger.info(f"[Retriever] 最终产出 {len(final_docs)} 个完整上下文 (总耗时 {time.time() - start_time:.4f}s)")
        return final_docs


# ===================================================
# 单例模式管理
# ===================================================
_global_retriever = None


def get_retriever() -> "AdvancedRetriever":
    """
    获取全局唯一的检索器实例 (单例模式)
    避免重复加载 Embedding 模型和数据库连接
    """
    global _global_retriever

    if _global_retriever is None:
        logger.info("[System] 初始化全局检索器...")
        _global_retriever = AdvancedRetriever(
            top_k=settings.TOP_K,
            recall_k=settings.RECALL_K,
            db_url=settings.DB_URL
        )
    return _global_retriever
