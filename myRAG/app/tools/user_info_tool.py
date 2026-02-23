"""
User Tools - 用户信息查询工具

使用 @tool 装饰器，简洁声明式。
通过 Spring Boot 内部 API 查询用户数据。
"""

import requests
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Spring Boot 内部 API 地址
API_BASE_URL = "http://localhost:5782/api/internal"


@tool
def get_user_info(user_id: int) -> str:
    """获取用户的详细信息。当用户询问账户余额、VIP等级、个人资料等问题时使用。输入 user_id，输出用户信息文本。"""
    try:
        logger.info(f"[get_user_info] 查询用户信息: user_id={user_id}")

        resp = requests.get(
            f"{API_BASE_URL}/user/info",
            params={"userId": user_id},
            timeout=5,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            return f"查询失败: {result.get('message', '未知错误')}"

        data = result.get("data", {})
        info_text = (
            f"用户信息查询结果:\n"
            f"- 用户ID: {data.get('id')}\n"
            f"- 姓名: {data.get('username')}\n"
            f"- 账号: {data.get('account')}\n"
            f"- VIP等级: {data.get('vipLevel')}\n"
            f"- 账户余额: {data.get('balance')} 元\n"
            f"- 账户状态: {'正常' if data.get('status') == 0 else '禁用'}\n"
            f"- 注册时间: {data.get('createTime')}"
        )

        logger.info(f"[get_user_info] 查询成功: username={data.get('username')}")
        return info_text

    except requests.exceptions.RequestException as e:
        logger.error(f"[get_user_info] 网络请求失败: {e}")
        return f"查询失败: 无法连接到用户服务 ({str(e)})"
    except Exception as e:
        logger.error(f"[get_user_info] 未知错误: {e}")
        return f"查询失败: {str(e)}"


@tool
def get_user_stats(user_id: int) -> str:
    """获取用户的统计信息。当用户询问订单数量、消费金额、登录时间等问题时使用。输入 user_id，输出统计信息文本。"""
    try:
        logger.info(f"[get_user_stats] 查询用户统计: user_id={user_id}")

        resp = requests.get(
            f"{API_BASE_URL}/user/stats",
            params={"userId": user_id},
            timeout=5,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            return f"查询失败: {result.get('message', '未知错误')}"

        data = result.get("data", {})
        stats_text = (
            f"用户统计信息:\n"
            f"- 总订单数: {data.get('totalOrders')} 单\n"
            f"- 总消费金额: {data.get('totalSpent')} 元\n"
            f"- 最后登录时间: {data.get('lastLoginTime')}"
        )

        logger.info(f"[get_user_stats] 查询成功")
        return stats_text

    except Exception as e:
        logger.error(f"[get_user_stats] 查询失败: {e}")
        return f"查询失败: {str(e)}"
