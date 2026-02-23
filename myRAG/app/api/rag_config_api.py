"""
RAG 参数配置 API

提供 RAG 核心参数的运行时读写 + RAGAS 评估触发
参数持久化到 data/rag_params.json, 与 .env 敏感配置分离
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG 配置"])

# 持久化路径
PARAMS_FILE = Path(settings.DATA_DIR) / "rag_params.json"
EVAL_DIR = Path(settings.DATA_DIR) / "evaluations"


# =========================================================
# Request / Response Models
# =========================================================

class RAGConfigResponse(BaseModel):
    """当前 RAG 参数"""
    chunk_size: int = Field(description="文本切分块大小 (仅影响新上传文档)")
    chunk_overlap: int = Field(description="切分重叠大小 (仅影响新上传文档)")
    top_k: int = Field(description="检索返回的父文档数量")
    rrf_k: int = Field(description="RRF 平滑指数: 越大排名越平滑, 越小头部权重越大")
    table_weight: float = Field(description="RRF 表格类型权重 (1.0=不加权)")
    image_weight: float = Field(description="RRF 图片类型权重 (1.0=不加权)")


class RAGConfigUpdate(BaseModel):
    """更新请求 (所有字段可选, 只更新传入的字段)"""
    chunk_size: Optional[int] = Field(None, ge=100, le=2000, description="文本切分块大小")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500, description="切分重叠大小")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="检索返回数量")
    rrf_k: Optional[int] = Field(None, ge=1, le=200, description="RRF 平滑指数")
    table_weight: Optional[float] = Field(None, ge=0.0, le=3.0, description="RRF 表格权重")
    image_weight: Optional[float] = Field(None, ge=0.0, le=3.0, description="RRF 图片权重")


class EvalResult(BaseModel):
    """RAGAS 评估结果"""
    timestamp: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    num_questions: int = 0
    details: list = []


# =========================================================
# 内部工具函数
# =========================================================

def _load_params() -> dict:
    """从持久化文件加载参数"""
    if PARAMS_FILE.exists():
        try:
            return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_params(params: dict):
    """保存参数到持久化文件"""
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _apply_params_to_settings(params: dict):
    """将加载的参数应用到 settings 单例"""
    for key in ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "RRF_K",
                "TABLE_WEIGHT", "IMAGE_WEIGHT"]:
        if key in params:
            setattr(settings, key, params[key])


# 启动时加载持久化参数
_saved = _load_params()
if _saved:
    _apply_params_to_settings(_saved)
    logger.info(f"[RAGConfig] 从 {PARAMS_FILE} 加载参数: {_saved}")


# =========================================================
# Admin 权限检查 (复用 vector_admin 的逻辑)
# =========================================================

async def require_admin(request: Request):
    """要求管理员权限"""
    role = getattr(request.state, 'user_role', None)
    if role != 1:
        raise HTTPException(status_code=403, detail="仅管理员可访问此接口")


# =========================================================
# API Endpoints
# =========================================================

@router.get("/config", response_model=RAGConfigResponse)
async def get_config():
    """获取当前 RAG 参数"""
    return RAGConfigResponse(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
        rrf_k=settings.RRF_K,
        table_weight=settings.TABLE_WEIGHT,
        image_weight=settings.IMAGE_WEIGHT,
    )


@router.put("/config", response_model=RAGConfigResponse)
async def update_config(req: RAGConfigUpdate, _admin=Depends(require_admin)):
    """
    更新 RAG 参数 (需 admin 权限)

    - 立即更新 settings 单例 (运行时生效)
    - 持久化到 rag_params.json (重启后恢复)
    - CHUNK_SIZE / CHUNK_OVERLAP 仅影响新上传文档
    """
    # 读取现有持久化参数
    params = _load_params()

    # 只更新传入的字段
    update_data = req.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="没有需要更新的参数")

    # 映射到 settings 字段名
    field_map = {
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
        "top_k": "TOP_K",
        "rrf_k": "RRF_K",
        "table_weight": "TABLE_WEIGHT",
        "image_weight": "IMAGE_WEIGHT",
    }

    for api_key, settings_key in field_map.items():
        if api_key in update_data:
            val = update_data[api_key]
            params[settings_key] = val
            setattr(settings, settings_key, val)
            logger.info(f"[RAGConfig] 更新 {settings_key} = {val}")

    # 持久化
    _save_params(params)

    return RAGConfigResponse(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
        rrf_k=settings.RRF_K,
        table_weight=settings.TABLE_WEIGHT,
        image_weight=settings.IMAGE_WEIGHT,
    )


@router.post("/evaluate", response_model=EvalResult)
async def run_evaluation(_admin=Depends(require_admin)):
    """
    触发 RAGAS 评估 (同步执行, 需 admin 权限)

    读取 data/evaluation_data.json, 对每条 question:
    1. 调用 retriever 获取 contexts
    2. 调用 LLM 生成 answer
    3. 使用 RAGAS 计算 4 个指标
    """
    try:
        from app.rag.evaluator import run_ragas_evaluation
        result = await run_ragas_evaluation()
        return result
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAGAS 未安装, 请执行: pip install ragas. Error: {e}"
        )
    except Exception as e:
        logger.error(f"[RAGConfig] 评估失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@router.get("/evaluate", response_model=EvalResult)
async def get_evaluation():
    """获取最近一次评估结果"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 找最新的评估文件
    eval_files = sorted(EVAL_DIR.glob("eval_*.json"), reverse=True)
    if not eval_files:
        raise HTTPException(status_code=404, detail="暂无评估结果")

    try:
        data = json.loads(eval_files[0].read_text(encoding="utf-8"))
        return EvalResult(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取评估结果失败: {e}")
