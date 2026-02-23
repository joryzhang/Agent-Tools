"""
RAGAS 评估模块

读取预定义的 QA 评估数据集, 使用 retriever 和 LLM 生成回答,
然后通过 RAGAS 框架计算 4 个核心指标:
- Faithfulness (忠诚度): 回答是否基于 context
- Answer Relevancy (答案相关性): 回答是否切题
- Context Precision (上下文精确度): 检索到的 context 是否有用
- Context Recall (上下文召回率): 是否检索到了足够的信息
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from app.config import settings

logger = logging.getLogger(__name__)

# 路径
EVAL_DATA_FILE = Path(settings.DATA_DIR) / "evaluation_data.json"
EVAL_OUTPUT_DIR = Path(settings.DATA_DIR) / "evaluations"


def _load_eval_dataset() -> List[Dict]:
    """加载评估数据集"""
    if not EVAL_DATA_FILE.exists():
        raise FileNotFoundError(f"评估数据集不存在: {EVAL_DATA_FILE}")

    data = json.loads(EVAL_DATA_FILE.read_text(encoding="utf-8"))
    logger.info(f"[Evaluator] 加载 {len(data)} 条评估数据")
    return data


async def run_ragas_evaluation() -> dict:
    """
    执行 RAGAS 评估

    1. 读取评估数据集
    2. 对每条 question 调用 retriever + LLM
    3. 使用 RAGAS 计算指标
    4. 持久化结果

    Returns:
        dict: 评估结果 (含各指标分数)
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas import EvaluationDataset, SingleTurnSample

    # 1. 加载数据集
    eval_data = _load_eval_dataset()

    # 2. 获取 retriever 和 LLM
    from app.rag.retriever import get_retriever
    from app.rag.utils import format_docs
    from models import get_ali_model_client

    retriever = get_retriever()
    llm = get_ali_model_client(temperature=0)

    # 3. 为每条问题生成 answer 和 contexts
    samples = []

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            # 检索
            docs = retriever.get_relevant_documents(question)
            contexts = [doc.page_content for doc in docs]

            # 生成回答
            context_text = format_docs(docs)
            prompt = f"基于以下信息回答问题:\n\n{context_text}\n\n问题: {question}"

            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            answer = response.content

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
            samples.append(sample)

            logger.info(f"[Evaluator] 处理完成: {question[:30]}...")

        except Exception as e:
            logger.error(f"[Evaluator] 问题处理失败: {question[:30]}... - {e}")
            continue

    if not samples:
        raise RuntimeError("所有评估问题都处理失败, 无法计算指标")

    # 4. RAGAS 评估
    logger.info(f"[Evaluator] 开始 RAGAS 评估, 样本数: {len(samples)}")

    dataset = EvaluationDataset(samples=samples)

    # 使用通义千问作为评估 LLM
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from models import get_tencent_embeddings

    evaluator_llm = LangchainLLMWrapper(get_ali_model_client(temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_tencent_embeddings())

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 5. 构建输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # EvaluationResult 支持 [] 访问, 取分数 dict
    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    eval_result = {
        "timestamp": timestamp,
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
        "context_precision": scores.get("context_precision"),
        "context_recall": scores.get("context_recall"),
        "num_questions": len(samples),
        "details": [],
    }

    # 详细结果
    df = result.to_pandas()
    for _, row in df.iterrows():
        eval_result["details"].append({
            "question": row.get("user_input", ""),
            "answer": row.get("response", ""),
            "faithfulness": row.get("faithfulness"),
            "answer_relevancy": row.get("answer_relevancy"),
            "context_precision": row.get("context_precision"),
            "context_recall": row.get("context_recall"),
        })

    # 6. 持久化
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = EVAL_OUTPUT_DIR / f"eval_{timestamp}.json"
    output_file.write_text(
        json.dumps(eval_result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"[Evaluator] 评估完成, 结果保存到: {output_file}")

    return eval_result
