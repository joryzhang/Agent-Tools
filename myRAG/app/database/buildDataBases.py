from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, text
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()


class FileRecord(Base):
    """原始文件记录表"""
    __tablename__ = 'files'
    __table_args__ = {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_general_ci'}

    file_id = Column(String(64), primary_key=True)  # 文件MD5
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512))
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'), default=datetime.datetime.now())
    last_update = Column(DateTime, default=datetime.datetime.now())

    # 与分片的关联关系
    chunks = relationship("ParentChunk", back_populates="file", cascade="all, delete-orphan")


class ParentChunk(Base):
    """父分片存储表"""
    __tablename__ = 'parent_chunks'
    __table_args__ = {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_general_ci'}

    chunk_id = Column(String(64), primary_key=True)  # UUID
    file_id = Column(String(64), ForeignKey('files.file_id'))
    content = Column(Text(4294967295))  # 对应 MySQL 的 LONGTEXT
    chunk_type = Column(String(20))  # text, table, image_desc
    page_num = Column(Integer)
    media_path = Column(String(512), nullable=True)  # 图片的本地相对路径

    file = relationship("FileRecord", back_populates="chunks")


class ParentOverride(Base):
    """
    父文档覆盖记录表

    当管理员编辑子 chunk 时，在父文档内定位并替换对应片段，
    生成一份完整的打过补丁的父文档副本存入此表。
    检索时优先读此表，原始 parent_chunks 永不修改。
    """
    __tablename__ = 'parent_overrides'
    __table_args__ = {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_general_ci'}

    parent_chunk_id = Column(String(64), ForeignKey('parent_chunks.chunk_id'), primary_key=True)
    override_content = Column(Text(4294967295))    # 打过补丁的完整父文档内容
    original_content = Column(Text(4294967295))    # 首次编辑前的原始内容快照
    edit_count = Column(Integer, default=1)         # 累积编辑次数
    edited_by = Column(Integer, nullable=True)      # 编辑人 user_id (来自 JWT)
    edited_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))

    parent_chunk = relationship("ParentChunk")
