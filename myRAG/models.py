# \RAG\models.py
# 可用模型列表，以及获得访问模型的客户端
# 实际使用时可以根据自己的实际情况调整
import langchain_openai

ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"  
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qwq-plus"  
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"  

DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"


TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME = "HUNYUAN_API_KEY"
TENCENT_HUNYUAN_URL = "https://api.hunyuan.cloud.tencent.com/v1"
TENCENT_HUNYUAN_TURBO_MODEL = "hunyuan-turbos-latest"
TENCENT_HUNYUAN_REASONER_MODEL = "hunyuan-t1-latest"
TENCENT_HUNYUAN_LONGCONTEXT_MODEL = "hunyuan-large-longcontext"
TENCENT_HUNYUAN_EMBEDDING_MODEL = "hunyuan-embedding"
TENCENT_SECRET_ID_OS_VAR_NAME = "Tencent_SecretId"
TENCENT_SECRET_KEY_OS_VAR_NAME = "Tencent_SecretKey"

BAICHUAN_API_KEY_OS_VAR_NAME = "Baichuan_API_Key"
BAICHUAN_EMBEDDING_MODEL = "Baichuan-Text-Embedding"
BAICHUAN_EMBEDDING_URL = "https://api.baichuan-ai.com/v1/embeddings"

import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import inspect
from langchain_community.embeddings import BaichuanTextEmbeddings, DashScopeEmbeddings, HunyuanEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank
from openai import OpenAI as RawOpenAI


def get_chat_open_model_client(model="gpt-3.5-turbo", temperature=0.7, verbose=False, debug=False):
    return ChatOpenAI(temperature=temperature, model=model, verbose=verbose, debug=debug,
                      api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech/v1")


def get_openai():
    return RawOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech/v1")


def get_openai_virtual_model_client():
    '''
    通过LangChain获得一个OpenAI的虚拟模型客户端
    :return: 虚拟模型客户端
    '''
    return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", base_url="https://api.chatanywhere.tech/v1")


def get_lc_model_client(api_key=os.getenv(TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME), base_url=TENCENT_HUNYUAN_URL
                        , model=TENCENT_HUNYUAN_TURBO_MODEL, temperature=0.7, verbose=False, debug=False):
    '''
    过LangChain获得指定平台和模型的客户端
    可以通过传入api_key，base_url，model，temperature四个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认平台和模型为腾讯混元，温度=0.7
    '''
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if (debug):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature},key：{api_key}")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=temperature)


def get_ali_model_client(model=ALI_TONGYI_MAX_MODEL, temperature: int = 0.7, verbose=False, debug=False):
    '''
    过LangChain获得阿里大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认模型为阿里百炼里的qwen-max-latest，温度=0.7
    '''
    return ChatOpenAI(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL, streaming=True,
                      model=model,
                      temperature=temperature, tiktoken_model_name="gpt-3.5-turbo")
    # return get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
    #                            , model=model, temperature=temperature, verbose=verbose, debug=debug)


def get_ds_model_client(model=DEEPSEEK_CHAT_MODEL, temperature=0.7, verbose=False, debug=False):
    '''
    过LangChain获得DeepSeek大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认模型为DeepSeek的deepseek-chat，温度=0.7
    '''
    return get_lc_model_client(api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME), base_url=DEEPSEEK_URL
                               , model=model, temperature=temperature, verbose=verbose, debug=debug)


def get_baichuan_embeddings():
    '''
    通过LangChain获得一个百川嵌入模型的实例，百川嵌入模型服务限流严重，
    大概有10~20%的概率访问报错
    :return: 百川嵌入模型的实例
    '''
    return BaichuanTextEmbeddings(
        api_key=os.getenv(BAICHUAN_API_KEY_OS_VAR_NAME)
    )


def get_ali_embeddings():
    '''
    通过LangChain获得一个阿里通义千问嵌入模型的实例
    :return: 阿里通义千问嵌入模型的实例，目前为text-embedding-v3
    '''
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )


def get_tencent_embeddings():
    '''
    通过LangChain获得一个腾讯嵌入模型的实例
    :return: 腾讯嵌入模型的实例
    '''
    return HunyuanEmbeddings(
        hunyuan_secret_id=os.getenv(TENCENT_SECRET_ID_OS_VAR_NAME),
        hunyuan_secret_key=os.getenv(TENCENT_SECRET_KEY_OS_VAR_NAME),
        region="ap-guangzhou",
    )


def get_ali_clients():
    '''
    产生阿里大模型客户端和嵌入模型的客户端
    :return: 阿里大模型客户端和嵌入模型的客户端
    '''
    return get_ali_model_client(), get_ali_embeddings()


def get_tencent_clients():
    '''
    产生腾讯的大模型客户端和嵌入模型的客户端
    :return: 腾讯的大模型客户端和腾讯嵌入模型的客户端
    '''
    return get_lc_model_client(), get_tencent_embeddings()


def get_a_t_mix_clients():
    '''
    混合产生阿里的大模型客户端和腾讯嵌入模型的客户端
    :return: 阿里的大模型客户端和腾讯嵌入模型的客户端
    '''
    return get_ali_model_client(), get_tencent_embeddings()


def get_ali_rerank(top_n=3):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    :return: 阿里通义千问嵌入模型的实例
    '''
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
    )
