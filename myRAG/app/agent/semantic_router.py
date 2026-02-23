"""
Semantic Router - 基于向量相似度的快速意图路由器

原理:
1. 预定义每个意图的示例句子 (utterances)
2. 启动时将所有示例编码为向量并缓存
3. 请求到达时, 将 query 编码为向量, 与所有示例做余弦相似度
4. 取最高分 route, 如果分数 ≥ threshold → 直接返回意图
5. 分数 < threshold → 返回 None (交给 LLM 兜底)

性能: ~50ms (仅 embedding API 调用 + numpy 计算)
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import logging
import asyncio

logger = logging.getLogger(__name__)


class SemanticRoute:
    """单个意图路由定义"""

    def __init__(self, name: str, utterances: List[str]):
        self.name = name
        self.utterances = utterances
        self.embeddings: Optional[np.ndarray] = None  # shape: (n, dim)


class SemanticClassification:
    """语义路由分类结果"""

    def __init__(self, intent: str, confidence: float, reasoning: str):
        self.intent = intent
        self.confidence = confidence
        self.reasoning = reasoning


class SemanticRouter:
    """
    基于向量相似度的快速意图路由器

    复用项目现有的腾讯 HunyuanEmbeddings, 零新增依赖。
    """

    def __init__(self, embedding_fn=None, threshold: float = 0.78):
        """
        Args:
            embedding_fn: LangChain Embeddings 实例 (需有 embed_documents / embed_query)
            threshold: 余弦相似度阈值, 低于此值交给 LLM 兜底
        """
        self.threshold = threshold

        if embedding_fn is None:
            from models import get_tencent_embeddings
            self.embedding_fn = get_tencent_embeddings()
        else:
            self.embedding_fn = embedding_fn

        # 定义意图路由
        self.routes = self._define_routes()

        # 缓存: 启动时编码所有示例句子
        self._initialized = False

    def _define_routes(self) -> List[SemanticRoute]:
        """
        定义意图路由和示例句子

        每个 route 包含 15-30 个典型表述, 覆盖:
        - 标准问法
        - 口语化表达
        - 简短/详细 变体
        """
        return [
            SemanticRoute(
                name="user_info",
                utterances=[
                    # 账户/余额
                    "我的余额是多少",
                    "查一下我的账户余额",
                    "我账户还有多少钱",
                    "余额查询",
                    "看看我还剩多少余额",
                    # VIP/等级
                    "我是VIP吗",
                    "我的会员等级是什么",
                    "查询我的VIP信息",
                    "我是什么级别的会员",
                    "我的等级",
                    # 个人信息
                    "查询我的个人信息",
                    "我的账户信息",
                    "看看我的资料",
                    "我的用户信息是什么",
                    # 订单/消费统计
                    "我有多少订单",
                    "查看我的订单数量",
                    "我消费了多少钱",
                    "我的消费金额是多少",
                    "查询我的消费统计",
                    "我的订单统计",
                    "我一共下了几个单",
                    "我的消费记录",
                ],
            ),
            SemanticRoute(
                name="knowledge",
                utterances=[
                    # 规则/政策
                    "违约金怎么算",
                    "违约金的计算方式",
                    "退款政策是什么",
                    "如何申请退款",
                    "退款流程",
                    # VIP权益
                    "VIP有什么特权",
                    "会员权益有哪些",
                    "VIP会员享受什么服务",
                    "VIP的优惠政策是什么",
                    # 合同/条款
                    "合同的免责条款是什么",
                    "服务条款说明",
                    "用户协议内容",
                    "免责声明是什么",
                    # 业务流程
                    "怎么开通VIP",
                    "如何升级会员",
                    "注册流程是什么",
                    "如何修改密码",
                    "怎么绑定手机号",
                    # 产品说明
                    "这个产品有什么功能",
                    "服务内容介绍",
                    "有哪些套餐",
                    "价格是多少",
                ],
            ),
            SemanticRoute(
                name="mixed",
                utterances=[
                    # 个人信息 + 知识库
                    "我是VIP吗 VIP有什么特权",
                    "我的余额够支付违约金吗",
                    "我的等级能享受什么优惠",
                    "我的消费金额达到VIP标准了吗",
                    "以我的会员等级可以申请退款吗",
                    "查一下我的余额 再告诉我退款流程",
                    "我是什么等级的会员 有什么专属权益",
                    "我的订单有没有违约 违约金怎么算的",
                ],
            ),
        ]

    async def initialize(self):
        """
        初始化: 将所有 route 示例句子编码为向量并缓存

        仅在首次调用时执行, 后续调用跳过。
        """
        if self._initialized:
            return

        logger.info("[SemanticRouter] 开始初始化, 编码示例句子...")

        for route in self.routes:
            try:
                # 使用 asyncio.to_thread 避免阻塞
                embeddings = await asyncio.to_thread(
                    self.embedding_fn.embed_documents, route.utterances
                )
                route.embeddings = np.array(embeddings, dtype=np.float32)

                logger.info(
                    f"[SemanticRouter] Route '{route.name}' 编码完成, "
                    f"{len(route.utterances)} 个示例, 维度: {route.embeddings.shape}"
                )
            except Exception as e:
                logger.error(
                    f"[SemanticRouter] Route '{route.name}' 编码失败: {e}",
                    exc_info=True,
                )
                route.embeddings = None

        self._initialized = True
        logger.info("[SemanticRouter] 初始化完成")

    async def classify(self, query: str) -> Optional[SemanticClassification]:
        """
        对 query 做语义匹配

        Returns:
            SemanticClassification 如果分数 ≥ threshold, 否则 None
        """
        # 确保已初始化
        await self.initialize()

        try:
            # 1. 编码 query
            query_embedding = await asyncio.to_thread(
                self.embedding_fn.embed_query, query
            )
            query_vec = np.array(query_embedding, dtype=np.float32)

            # 2. 与所有 route 做余弦相似度
            best_route: Optional[str] = None
            best_score: float = -1.0
            all_scores: Dict[str, float] = {}

            for route in self.routes:
                if route.embeddings is None:
                    continue

                # 余弦相似度: dot(q, r) / (|q| * |r|)
                similarities = self._cosine_similarity(query_vec, route.embeddings)

                # 取该 route 下所有示例的最高分
                max_sim = float(np.max(similarities))
                all_scores[route.name] = round(max_sim, 4)

                if max_sim > best_score:
                    best_score = max_sim
                    best_route = route.name

            logger.info(
                f"[SemanticRouter] query='{query}' | "
                f"scores={all_scores} | best={best_route}({best_score:.4f}) | "
                f"threshold={self.threshold}"
            )

            # 3. 分数判定
            if best_route and best_score >= self.threshold:
                return SemanticClassification(
                    intent=best_route,
                    confidence=best_score,
                    reasoning=f"语义匹配命中 (score={best_score:.4f}, threshold={self.threshold})",
                )

            # 未达阈值, 返回 None 交给 LLM 兜底
            logger.info(
                f"[SemanticRouter] 未达阈值, 交给 LLM 兜底 "
                f"(best={best_score:.4f} < threshold={self.threshold})"
            )
            return None

        except Exception as e:
            logger.error(f"[SemanticRouter] 分类失败: {e}", exc_info=True)
            return None

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        计算 query 向量与矩阵中每行的余弦相似度

        Args:
            query: shape (dim,)
            matrix: shape (n, dim)

        Returns:
            shape (n,) 的相似度数组
        """
        # 归一化
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        matrix_norms = matrix / (
            np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        )
        # 点积 = 余弦相似度 (归一化后)
        return matrix_norms @ query_norm
