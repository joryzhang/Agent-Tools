import os
import logging
from parse.rapid_parser import RapidPDFParser

from app.database.storage import SQLAlchemyDocStore

from preRetrive.processor import ParentChildProcessor

from app.database.VectorStoreManager import VectorStoreManager

from app.config import settings

logger = logging.getLogger(__name__)


# 流程编排


class RAGPipelineOrchestrator:
    def __init__(self):
        # 初始化所有组件
        self.parser = RapidPDFParser()
        self.db_store = SQLAlchemyDocStore(settings.DB_URL)  # MySQL 持久化
        self.processor = ParentChildProcessor()
        self.vector_db = VectorStoreManager()

    async def run_pipeline(self, pdf_path: str):
        """
        一键完成：解析 -> 增强 -> 切分 -> 双入库
        """
        file_name = os.path.basename(pdf_path)
        file_hash = self.parser._get_file_hash(pdf_path)

        # 1. 解析基础内容 (同步)
        logger.info(f"[编排] 开始解析文档: {file_name}")
        parse_result = self.parser.parse_pdf(pdf_path, file_name=file_name)

        # 2. 异步处理图片语义增强 (异步)
        image_docs = []
        if parse_result.get("saved_images"):
            image_docs = await self.parser.process_images(parse_result["metadata"], parse_result["saved_images"])

        # 3. 转换为基础 Document 对象并进行语义切分
        # 文本和表格先转为 Document
        text_table_docs = self.parser.to_documents(parse_result)

        all_raw_docs = text_table_docs + image_docs

        # 调用预处理器生成父子索引
        parent_docs, child_docs = self.processor.process(all_raw_docs)

        # 4. 双入库 (企业级持久化)
        # A. 父块存入 MySQL (SQLAlchemy)
        file_info = {
            'file_hash': file_hash,
            'file_name': file_name,
            'file_path': pdf_path
        }
        self.db_store.save_parsed_data(file_info, parent_docs)

        # B. 子块存入 ChromaDB
        self.vector_db.add_documents(child_docs)

        logger.info(f"[编排] 文档 {file_name} 处理完成")
        return file_hash
