import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# =========================================================
# 基础路径配置
# =========================================================
# 获取当前文件的父目录的父目录，即项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    """
    系统全局配置类
    基于 Pydantic，自动从环境变量或 .env 文件中加载
    """
    # --- 1. 项目基础信息 ---
    PROJECT_NAME: str = "Enterprise RAG System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False  # 生产环境务必改为 False
    ALLOWED_HOSTS: list[str]

    # --- 1.5 安全认证配置 ---
    # 必须与 Spring Boot 端的 jwt.secret 保持完全一致
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"

    # --- 2. LLM 模型配置 (OpenAI 兼容接口) ---
    # 你的 API Key (必填，不配报错)
    OPENAI_API_KEY: str
    # API 代理地址 (如果你用 DeepSeek, OneAPI 或中转站，这里必须配)
    # 默认值设为官方地址，但通常我们需要改
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    # 模型名称
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"
    # 生成温度 (0 为最精确，1 为最有创造力)
    LLM_TEMPERATURE: float = 0.0

    # --- 3. 向量与 Embedding 配置 ---
    # 腾讯 Embedding 或者其他厂商
    EMBEDDING_MODEL_NAME: str = "tencent-embedding"
    # 向量维度 (需与模型匹配)
    EMBEDDING_DIM: int = 1024

    # --- 4. 数据库配置 ---
    # MySQL 连接字符串 (必填)
    # 格式: mysql+mysqldb://user:pass@host:port/dbname?charset=utf8mb4
    DB_URL: str
    # 是否打印 SQL 语句 (调试用)
    ECHO_SQL: bool = False

    # --- 5. 文件存储路径配置 (自动拼接) ---
    # 所有数据存储的根目录
    DATA_DIR: Path = BASE_DIR / "data"


    # 向量数据库持久化目录
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

    # PDF 解析出的图片存储目录
    IMG_OUTPUT_DIR: Path = DATA_DIR / "images"

    # 缓存目录 (存放解析后的 JSON 结果)
    CACHE_DIR: Path = DATA_DIR / "cache"

    # --- 6. RAG 核心参数 ---
    # 文本切片大小
    CHUNK_SIZE: int = 500
    # 切片重叠大小
    CHUNK_OVERLAP: int = 50
    # 检索返回数量
    TOP_K: int = 2
    # RRF 融合参数
    RRF_K: int = 60
    # RRF 表格权重 (1.0 = 不加权)
    TABLE_WEIGHT: float = 1.0
    # RRF 图片权重 (1.0 = 不加权)
    IMAGE_WEIGHT: float = 1.0
    # 每路检索器的子切片召回数量 (内部工程参数, 不暴露给前端)
    # 该值应远大于 TOP_K, 为 RRF 融合提供足够候选
    RECALL_K: int = 10
    # SemanticRouter 意图路由阈值 (低于此分数 fallback 到 LLM 分类)
    SEMANTIC_THRESHOLD: float = 0.78
    # RAG 上下文 token 预算
    # 模型 context window 中分配给检索文档的最大 token 数
    # 剩余空间留给: system prompt (~300) + chat history (≤4000) + output
    RAG_CONTEXT_MAX_TOKENS: int = 5000

    # --- Pydantic 配置 ---
    # 指定环境变量文件为 .env，编码 utf-8
    # extra="ignore" 表示如果 .env 里有多余的字段，不报错，直接忽略
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# 实例化配置对象 (单例模式)
# 其他文件直接 import settings 即可使用
settings = Settings()


# =========================================================
# 自动创建必要目录
# =========================================================
import logging as _logging

_config_logger = _logging.getLogger(__name__)


def ensure_directories():
    """在模块被导入时，自动检查并创建必要文件夹"""
    directories = [
        settings.DATA_DIR,
        settings.CHROMA_DB_DIR,
        settings.IMG_OUTPUT_DIR,
        settings.CACHE_DIR
    ]
    for directory in directories:
        if not directory.exists():
            _config_logger.info(f"[Config] 自动创建目录: {directory}")
            directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
