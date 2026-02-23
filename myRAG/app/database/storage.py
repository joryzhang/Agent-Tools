import os
import uuid
import logging
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from app.database.buildDataBases import Base, FileRecord, ParentChunk, ParentOverride
except ImportError:
    from app.database.buildDataBases import Base, FileRecord, ParentChunk, ParentOverride

load_dotenv()

logger = logging.getLogger(__name__)


class SQLAlchemyDocStore:
    def __init__(self, db_uri):
        """
        初始化连接
        db_uri: 数据库连接字符串，如果不传则从环境变量 MYSQL_URI 读取
        """
        if not db_uri:
            db_uri = os.getenv("MYSQL_URI", "")

        if not db_uri:
            raise ValueError("必须提供 db_uri 或设置 MYSQL_URI 环境变量")

        self.engine = create_engine(db_uri, pool_recycle=3600, pool_size=10, echo=False)
        Base.metadata.create_all(self.engine)  # 自动建表（如果不存在）
        self.Session = sessionmaker(bind=self.engine)

    def save_parsed_data(self, file_info: dict, parent_docs: list):
        """
        将预处理后的父分片存入 MySQL
        file_info: dict, 包含 'hash', 'name', 'path'
        parent_docs: list[Document], LangChain Document 对象列表
        """
        session = self.Session()
        try:
            # 1. 插入或更新文件记录
            file_record = FileRecord(
                file_id=file_info['file_hash'],
                file_name=file_info['file_name'],
                file_path=file_info.get('file_path', '')
            )
            session.merge(file_record)

            # 2. 先删除该文件已有的旧分片,保证数据与解析结果一致
            session.query(ParentChunk).filter(ParentChunk.file_id == file_info['file_hash']).delete()

            new_chunks = []
            for doc in parent_docs:
                c_id = doc.metadata.get('doc_id')
                if not c_id:
                    c_id = str(uuid.uuid4())

                chunk_type = doc.metadata.get('type', 'text')
                page_num = doc.metadata.get('page', 0)
                media_path = doc.metadata.get('image_path', '')
                if not media_path:
                    media_path = doc.metadata.get('source', '') if chunk_type == 'image' else ''

                chunk = ParentChunk(
                    chunk_id=c_id,
                    file_id=file_info['file_hash'],
                    content=doc.page_content,
                    chunk_type=chunk_type,
                    page_num=page_num,
                    media_path=media_path
                )
                new_chunks.append(chunk)

            session.add_all(new_chunks)
            session.commit()
            logger.info(f"[MySQL] 入库成功: 文件 {file_info['file_name']} ({len(new_chunks)} 个分片)")
        except Exception as e:
            session.rollback()
            logger.error(f"[MySQL] 入库失败: {e}")
            raise
        finally:
            session.close()

    # =========================================================
    # 检索时调用
    # =========================================================

    def get_parent_by_id(self, chunk_id: str) -> Document:
        """
        检索时调用：根据 parent chunk ID 找回原文。
        优先读 override 表（如果有人工编辑过），否则回退到原始 parent_chunks。
        """
        session = self.Session()
        try:
            # 1. 优先查 override 表
            override = session.query(ParentOverride).filter(
                ParentOverride.parent_chunk_id == chunk_id
            ).first()

            if override:
                # 用打过补丁的内容
                chunk = session.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                return Document(
                    page_content=override.override_content,
                    metadata={
                        "page_num": chunk.page_num if chunk else 0,
                        "chunk_type": chunk.chunk_type if chunk else "text",
                        "media_path": chunk.media_path if chunk else "",
                        "has_override": True,
                        "edit_count": override.edit_count,
                    }
                )

            # 2. 回退到原始
            chunk = session.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
            if not chunk:
                return None

            return Document(
                page_content=chunk.content,
                metadata={
                    "page_num": chunk.page_num,
                    "chunk_type": chunk.chunk_type,
                    "media_path": chunk.media_path,
                }
            )
        except Exception as e:
            logger.error(f"[MySQL] 检索失败: {e}")
            return None
        finally:
            session.close()

    # =========================================================
    # Override 管理（供 vector_admin 调用）
    # =========================================================

    def get_parent_content_for_patch(self, parent_id: str) -> str | None:
        """
        获取打补丁的基准文本。
        如果已有 override → 用 override_content（这样多次编辑可以累积）。
        否则 → 用原始 parent_chunks.content。
        """
        session = self.Session()
        try:
            override = session.query(ParentOverride).filter(
                ParentOverride.parent_chunk_id == parent_id
            ).first()
            if override:
                return override.override_content

            parent = session.query(ParentChunk).filter(
                ParentChunk.chunk_id == parent_id
            ).first()
            return parent.content if parent else None
        finally:
            session.close()

    def get_raw_parent_content(self, parent_id: str) -> str | None:
        """获取原始（未打补丁）的父文档内容，用于首次生成 override 时保存快照。"""
        session = self.Session()
        try:
            parent = session.query(ParentChunk).filter(
                ParentChunk.chunk_id == parent_id
            ).first()
            return parent.content if parent else None
        finally:
            session.close()

    def upsert_override(self, parent_id: str, override_content: str,
                        original_content: str, user_id: int = None):
        """
        创建或更新父文档覆盖记录。
        - 首次编辑：新建记录，保存 original_content 快照
        - 后续编辑：更新 override_content，递增 edit_count
        """
        session = self.Session()
        try:
            existing = session.query(ParentOverride).filter(
                ParentOverride.parent_chunk_id == parent_id
            ).first()

            if existing:
                existing.override_content = override_content
                existing.edit_count += 1
                existing.edited_by = user_id
                existing.edited_at = datetime.datetime.now()
            else:
                record = ParentOverride(
                    parent_chunk_id=parent_id,
                    override_content=override_content,
                    original_content=original_content,
                    edit_count=1,
                    edited_by=user_id,
                )
                session.add(record)

            session.commit()
            logger.info(f"[Override] 父文档覆盖已更新: {parent_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"[Override] 覆盖更新失败: {e}")
            raise
        finally:
            session.close()

    def delete_override(self, parent_id: str) -> bool:
        """删除 override 记录，回滚到原始父文档内容。"""
        session = self.Session()
        try:
            deleted = session.query(ParentOverride).filter(
                ParentOverride.parent_chunk_id == parent_id
            ).delete()
            session.commit()
            return deleted > 0
        except Exception as e:
            session.rollback()
            logger.error(f"[Override] 删除失败: {e}")
            return False
        finally:
            session.close()
