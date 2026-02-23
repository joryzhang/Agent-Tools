# RAG 系统提示词模版
# 统一管理，方便后续支持多版本或从数据库/前端动态加载

from langchain_core.prompts import ChatPromptTemplate

# RAG 标准 System Prompt
RAG_SYSTEM_PROMPT_TEMPLATE = """你是一个专业的企业级文档助手。你的任务是根据提供的【上下文信息】回答用户的问题。

严守以下规则：
1. 你的回答必须完全基于提供的【上下文信息】。
2. 如果【上下文信息】中没有包含问题的答案，你必须直接回答“根据已有文档，我无法回答这个问题”，严禁编造答案。
3. 回答要专业、简洁、逻辑清晰。
4. 请用中文回答。
"""

# 构建 ChatPromptTemplate 的工厂函数
def get_rag_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT_TEMPLATE),
        ("human", """
【上下文信息】：
{context}

【用户问题】：
{question}
""")
    ])
