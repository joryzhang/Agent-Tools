"""
认证中间件 (Hybrid Auth Pattern)
策略: Session 主权 (Spring Boot), Token 通行 (Python)

本模块只负责无状态的 JWT 验签。
"""

import jwt
import logging
import os
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings

logger = logging.getLogger(__name__)

# 不需要认证的路径
PUBLIC_PATHS = [
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
]

# 开发模式开关 (从环境变量读取，默认为 False)
# 注意：生产环境 .env 中必须确保 DEV_MODE=False 或者不配置
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"


class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT 认证中间件
    
    职责:
    1. 拦截所有请求
    2. 检查 Authorization Header (Bearer Token)
    3. 使用共享密钥 (settings.JWT_SECRET) 验签
    4. 将用户信息注入 request.state
    """
    
    async def dispatch(self, request: Request, call_next):
        
        # 0. 放行 OPTIONS 请求 (CORS 预检)
        if request.method == "OPTIONS":
            return await call_next(request)

        # 1. 放行公开路径
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # 2. 开发模式兜底 (仅限本地调试)
        if DEV_MODE:
            request.state.user_id = 1
            request.state.username = "dev_user"
            request.state.user_role = "admin"
            # logger.debug("[Auth] 开发模式: 已跳过认证")
            return await call_next(request)
        
        # 3. 执行 JWT 认证
        user_id = self._verify_jwt(request)
        
        if user_id:
            # 认证成功
            request.state.user_id = user_id
            return await call_next(request)
        else:
            # 认证失败 (401)
            return JSONResponse(
                status_code=401,
                content={
                    "code": 401,
                    "message": "Authentication Failed: Invalid or missing token",
                    "data": None
                }
            )
    
    def _is_public_path(self, path: str) -> bool:
        return any(path.startswith(public_path) for public_path in PUBLIC_PATHS)
    
    def _verify_jwt(self, request: Request) -> Optional[int]:
        """
        验证 JWT Token
        """
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.split(" ")[1]
            
            # 使用配置中的密钥验签
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # 提取数据
            # 假设 Spring Boot 生成的 subject 是 string 类型的 user_id
            user_id = int(payload.get("sub"))
            role = payload.get("role", "user")
            
            # 注入状态
            request.state.user_role = role
            
            return user_id
            
        except jwt.ExpiredSignatureError:
            logger.warning("[Auth] Token 已过期")
        except jwt.InvalidTokenError as e:
            logger.warning(f"[Auth] Token 无效: {e}")
        except Exception as e:
            logger.error(f"[Auth] 认证异常: {e}")
            
        return None


def get_current_user_id(request: Request) -> int:
    """
    依赖注入辅助函数
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="未登录")
    return user_id
