import json
import time
import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 引入自定义模块
from app.config import settings
from app.ingest.service import IngestionService
from app.schemas import ChatRequest
from app.rag.retriever import AdvancedRetriever
from app.agent.executor import RAGAgent
from app.middleware.auth import AuthMiddleware

# TTL 缓存: Agent 实例最多 100 个, 1 小时过期
from cachetools import TTLCache

# =========================================================
# 日志配置 (带颜色: INFO=白, WARNING=黄, ERROR=红)
# =========================================================
import os as _os
import sys as _sys

# Windows 终端启用 ANSI 转义码支持
if _sys.platform == "win32":
    _os.system("")


class _ColoredFormatter(logging.Formatter):
    """
    自定义日志 Formatter，按级别着色
    - DEBUG:    青色 (Cyan)
    - INFO:     白色 (默认)
    - WARNING:  黄色
    - ERROR:    红色
    - CRITICAL: 红色加粗
    """
    _RESET = "\033[0m"
    _COLORS = {
        logging.DEBUG:    "\033[36m",      # Cyan
        logging.INFO:     "\033[37m",      # White
        logging.WARNING:  "\033[33m",      # Yellow
        logging.ERROR:    "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m",    # Bold Red
    }

    def format(self, record):
        color = self._COLORS.get(record.levelno, self._RESET)
        message = super().format(record)
        return f"{color}{message}{self._RESET}"


_handler = logging.StreamHandler()
_handler.setFormatter(_ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
# force=True: 移除已有 handler, 用我们的彩色 handler 替换 (而非叠加)
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)

logger = logging.getLogger(__name__)


# =========================================================
# 1. 全局状态管理 (Global State)
# =========================================================
class GlobalState:
    retriever: AdvancedRetriever = None
    # Agent 缓存 (TTL LRU: 最多 100 个, 1 小时过期)
    agent_cache = TTLCache(maxsize=100, ttl=3600) if TTLCache else {}


state = GlobalState()


# =========================================================
# 2. 生命周期管理 (Lifespan Events)
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[System] 正在启动 RAG 引擎...")
    start_time = time.time()

    try:

        # A. 初始化检索器 (使用单例)
        from app.rag.retriever import get_retriever
        state.retriever = get_retriever()
        logger.info("[System] 检索器加载完成")

        # B. 初始化 Agent 缓存
        state.agent_cache = TTLCache(maxsize=100, ttl=3600) if TTLCache else {}
        logger.info("[System] Agent TTL 缓存初始化完成 (maxsize=100, ttl=3600s)")

        # C. 预热 SemanticRouter 向量缓存 (避免首次请求延迟)
        try:
            from app.agent.executor import warmup_router
            await warmup_router()
            logger.info("[System] SemanticRouter 预热完成")
        except Exception as e:
            logger.warning(f"[System] SemanticRouter 预热失败 (不影响启动): {e}")

        logger.info(f"[System] 服务启动成功！总耗时: {time.time() - start_time:.2f}s")
        yield

    except Exception as e:
        logger.error(f"[System] 启动失败: {e}")
        raise e
    finally:
        logger.info("[System] 服务关闭，清理资源...")


# =========================================================
# 3. FastAPI 应用初始化
# =========================================================
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    description="基于 LangGraph Agent 的企业级 RAG 服务"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置认证中间件
app.add_middleware(AuthMiddleware)

# 注册向量库管理 API
from app.api.vector_admin import router as vector_admin_router
app.include_router(vector_admin_router)

from app.api.rag_config_api import router as rag_config_router
app.include_router(rag_config_router)


# =========================================================
# 4. 辅助函数
# =========================================================


def get_or_create_agent(user_id: int) -> RAGAgent:
    """
    获取或创建 Agent (带缓存)
    
    Args:
        user_id: 用户 ID
        
    Returns:
        RAGAgent 实例
    """
    if user_id not in state.agent_cache:
        logger.info(f"[Agent] 为用户 {user_id} 创建新 Agent")
        state.agent_cache[user_id] = RAGAgent(user_id=user_id)
    return state.agent_cache[user_id]


def format_sse(data: dict) -> str:
    """格式化为 SSE (Server-Sent Events) 标准字符串"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# =========================================================
# 5. API 接口定义
# =========================================================

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "code": 0,
        "message": "success",
        "data": {
            "status": "healthy",
            "modules": {
                "retriever": state.retriever is not None,
                "agent_cache_size": len(state.agent_cache)
            }
        }
    }


@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(..., description="上传的PDF文件")):
    logger.info(f"[Upload] 文件上传: {file.filename}")
    service = IngestionService()
    response_result = await service.process_file(file)
    result = response_result.get("status") == "success"

    # 重新加载 BM25 索引
    if state.retriever:
        state.retriever.reload_bm25()

    if result:
        return {
            "code": 0,
            "message": "文件上传并处理成功",
            "data": {
                "filename": file.filename,
                "success": True
            }
        }
    else:
        # 上传失败的情况
        return {
            "code": 500,
            "message": f"处理失败: {response_result.get('message', '未知错误')}",
            "data": None
        }


@app.post("/api/v1/chat/stream")
async def chat_stream(
        chat_request: ChatRequest,
        request: Request,
):
    """
    流式对话接口 (Agent 版本)
    
    功能:
    - 自动意图识别 (在 Agent 内部)
    - 多工具调用 (用户信息 + 知识库)
    - 流式响应
    
    返回类型: text/event-stream
    """
    logger.info(f"[Request] 用户提问: {chat_request.query}")

    # 从认证中间件获取 user_id
    if not hasattr(request.state, "user_id"):
        logger.warning(f"[Security] 未授权访问尝试: {request.client.host}")
        raise HTTPException(status_code=401,
                            detail="Authentication required: User identity not found in request state.")
    user_id = getattr(request.state, "user_id", 1)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user identity.")
    logger.info(f"[Request] 用户ID: {user_id}")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. 发送开始信号
            yield format_sse({"event": "start"})
            logger.info(f"[Server] 发送开始信号")

            # 2. 获取或创建 Agent
            agent = get_or_create_agent(user_id)
            logger.info(f"[Server] Agent 获取成功")

            # 3. 流式执行 Agent (包含意图识别)
            chunk_count = 0

            logger.info(f"[Server] 开始流式执行 Agent")

            async for chunk in agent.astream(chat_request.query, chat_history=chat_request.history, temperature=chat_request.temperature):
                chunk_count += 1

                if isinstance(chunk, dict):
                    msg_type = chunk.get("type")
                    data = chunk.get("data")

                    if msg_type == "intent":
                        # 格式化 intent 为 thinking 文本，保持前端兼容性
                        intent_str = f"""🤔 意图识别结果:
类型: {data.get('intent')}
置信度: {data.get('confidence', 0):.2f}
理由: {data.get('reasoning')}"""
                        yield format_sse({"thinking": intent_str})
                        logger.info(f"[Server] 发送 thinking")

                    elif msg_type == "content":
                        # 发送流式内容
                        yield format_sse({"content": data})

                    elif msg_type == "error":
                        logger.error(f"[Server] Agent 错误: {data}")
                        yield format_sse({"content": f"\n\n[系统错误]: {data}"})

                else:
                    # 以前的兼容逻辑 (如果有些 chunk 还是 string)
                    yield format_sse({"content": str(chunk)})

            # 4. 发送结束信号
            yield format_sse({"event": "done"})
            yield "data: [DONE]\n\n"
            logger.info(f"[Server] 发送结束信号")

        except Exception as e:
            logger.error(f"[Server] 生成中断: {e}", exc_info=True)
            yield format_sse({"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )



# 开发模式启动入口
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
