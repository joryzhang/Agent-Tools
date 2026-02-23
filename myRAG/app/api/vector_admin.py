"""
Vector Admin API - 向量库管理接口
提供向量条目的查看、更新、删除功能

更新 chunk 时，会在父文档中定位并替换对应片段，
生成打过补丁的完整父文档副本存入 parent_overrides 表。
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from pydantic import BaseModel
import logging
from app.rag.retriever import get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vectors", tags=["向量库管理"])


# =========================================================
# Request / Response Models
# =========================================================

class VectorItem(BaseModel):
    """单条向量条目"""
    id: str
    content: str
    metadata: dict


class VectorListResponse(BaseModel):
    """向量列表响应"""
    total: int
    items: list[VectorItem]


class VectorUpdateRequest(BaseModel):
    """更新请求"""
    content: str


# =========================================================
# 内部工具函数
# =========================================================

def _get_chroma_collection():
    """获取 ChromaDB collection manager"""
    from app.database.VectorStoreManager import VectorStoreManager
    manager = VectorStoreManager()
    return manager


def _get_sql_store():
    """获取 SQLAlchemy doc store (复用 retriever 的实例)"""
    try:
        retriever = get_retriever()
        return retriever.sql_store
    except Exception:
        # 回退：直接创建新实例
        from app.database.storage import SQLAlchemyDocStore
        from app.config import settings
        return SQLAlchemyDocStore(db_uri=settings.DB_URL)


# =========================================================
# Admin 权限检查
# =========================================================

async def require_admin(request: Request):
    """要求管理员权限"""
    role = getattr(request.state, 'user_role', None)
    if role != 1:
        raise HTTPException(status_code=403, detail="仅管理员可访问此接口")


# =========================================================
# API Endpoints
# =========================================================

@router.get("", response_model=VectorListResponse)
async def list_vectors(
        keyword: Optional[str] = Query(None, description="按内容关键词过滤"),
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页条数"),
        _admin=Depends(require_admin),
):
    """获取向量库中所有条目（分页）"""
    try:
        manager = _get_chroma_collection()
        all_data = manager.vector_db.get()

        ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        items = []
        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            if keyword and keyword.lower() not in content.lower():
                continue

            items.append(VectorItem(
                id=doc_id,
                content=content,
                metadata=metadata or {}
            ))

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        paged_items = items[start:end]

        return VectorListResponse(total=total, items=paged_items)

    except Exception as e:
        logger.error(f"[VectorAdmin] 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询向量库失败: {str(e)}")


@router.put("/{vector_id}")
async def update_vector(
        vector_id: str,
        req: VectorUpdateRequest,
        request: Request,
        _admin=Depends(require_admin),
):
    """
    根据 ID 更新某条向量的文本内容。

    核心逻辑：
    1. 读取旧 chunk 内容和 parent_id
    2. 获取父文档基准文本（优先 override，否则原始）
    3. 在基准文本中定位旧 chunk → 替换为新内容 → 生成 patched 父文档
    4. 存入 parent_overrides 表
    5. 更新 ChromaDB 中的 chunk 内容 + embedding
    6. 热更新 BM25 索引
    """
    try:
        manager = _get_chroma_collection()
        collection = manager.vector_db._collection

        # 1. 验证 ID 是否存在，获取旧内容
        existing = collection.get(ids=[vector_id])
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail=f"向量 ID '{vector_id}' 不存在")

        old_content = existing["documents"][0]
        old_metadata = existing["metadatas"][0] if existing["metadatas"] else {}
        parent_id = old_metadata.get("parent_id")

        # 2. 父文档级补丁逻辑
        if parent_id:
            try:
                sql_store = _get_sql_store()

                # 获取打补丁的基准文本（如果已有 override 就基于 override 累积编辑）
                base_content = sql_store.get_parent_content_for_patch(parent_id)

                if base_content:
                    # 在基准文本中定位旧 chunk 内容 → 替换为新内容
                    if old_content in base_content:
                        patched_content = base_content.replace(old_content, req.content, 1)
                    else:
                        # 旧内容不在基准中 (可能编码差异或已被之前的编辑改过)
                        # 安全兜底：追加标注
                        logger.warning(
                            f"[VectorAdmin] 旧 chunk 内容未在父文档中精确匹配，"
                            f"采用追加模式。vector_id={vector_id}, parent_id={parent_id}"
                        )
                        patched_content = base_content + f"\n\n[chunk {vector_id[:8]} 已修正]\n{req.content}"

                    # 获取原始（未打补丁）的父文档内容，用于首次快照
                    raw_original = sql_store.get_raw_parent_content(parent_id)

                    # UPSERT override
                    user_id = getattr(request.state, 'user_id', None)
                    sql_store.upsert_override(
                        parent_id=parent_id,
                        override_content=patched_content,
                        original_content=raw_original or "",
                        user_id=user_id,
                    )
                    logger.info(f"[VectorAdmin] 父文档补丁已生成: parent_id={parent_id}")
                else:
                    logger.warning(f"[VectorAdmin] 未找到父文档 {parent_id}，跳过覆盖层生成")

            except Exception as e:
                # Override 写入失败不应阻塞 chunk 本身的更新
                logger.error(f"[VectorAdmin] Override 生成失败 (非致命): {e}", exc_info=True)

        # 3. 更新 ChromaDB chunk 本身
        new_embedding = manager.embedding_fn.embed_documents([req.content])[0]

        # 更新 metadata 打标记
        updated_metadata = {**old_metadata, "has_override": True}

        collection.update(
            ids=[vector_id],
            documents=[req.content],
            embeddings=[new_embedding],
            metadatas=[updated_metadata],
        )

        # 4. 热更新 BM25 索引
        try:
            from app.rag.retriever import get_retriever
            retriever = get_retriever()
            retriever.reload_bm25()
        except Exception as e:
            logger.warning(f"[VectorAdmin] BM25 热更新失败 (非致命): {e}")

        logger.info(f"[VectorAdmin] 更新成功: {vector_id}")
        return {"code": 0, "message": "更新成功", "id": vector_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VectorAdmin] 更新失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@router.delete("/{vector_id}")
async def delete_vector(
        vector_id: str,
        _admin=Depends(require_admin),
):
    """根据 ID 删除某条向量"""
    try:
        manager = _get_chroma_collection()
        collection = manager.vector_db._collection

        existing = collection.get(ids=[vector_id])
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail=f"向量 ID '{vector_id}' 不存在")

        collection.delete(ids=[vector_id])

        # 热更新 BM25 索引
        try:
            from app.rag.retriever import get_retriever
            retriever = get_retriever()
            retriever.reload_bm25()
        except Exception as e:
            logger.warning(f"[VectorAdmin] BM25 热更新失败 (非致命): {e}")

        logger.info(f"[VectorAdmin] 删除成功: {vector_id}")
        return {"code": 0, "message": "删除成功", "id": vector_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VectorAdmin] 删除失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
