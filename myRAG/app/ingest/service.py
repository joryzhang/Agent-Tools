"""
“文件上传”、“调用你的解析器”、“图片分析”、“切片”、“存数据库”
"""
import shutil
import uuid
import logging

from typing import Dict, Any
import os
from fastapi import UploadFile

from app.database.orchestrator import RAGPipelineOrchestrator
from app.config import settings

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):

        # 2. 初始化解析器 (你的 RapidPDFParser)
        self.orchestrator = RAGPipelineOrchestrator()

    async def process_file(self, file: UploadFile) -> Dict[str, Any]:
        # --- 1. 文件落地 (I/O) --- 倒序按照.分割拿到文件后缀
        file_ext = file.filename.split(".")[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_ext}"  # 生成唯一文件名
        saved_path = settings.DATA_DIR / "uploads" / unique_filename
        saved_path.parent.mkdir(parents=True, exist_ok=True)  # parent自动生成上层文件夹
        try:
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)  # 用“流式方式”把上传文件安全地写入磁盘

            logger.info(f"[Ingest] 文件已接收: {file.filename} -> {saved_path}")

            # 解析文件
            file_hash = await self.orchestrator.run_pipeline(str(saved_path))
            return {
                "filename": file.filename,
                "file_hash": file_hash,
                "status": "success",
                "message": "Pipeline execution completed"
            }
        except Exception as e:
            logger.error(f"[Service] 处理失败: {e}")
            # 出错后清理临时文件
            if saved_path.exists():
                os.remove(saved_path)
            raise e
        finally:
            # 这里的策略取决于你：
            # 如果你想保留原始 PDF 做备份，就别删。
            # 如果你想省空间，可以在这里 os.remove(saved_path)
            os.remove(saved_path)
