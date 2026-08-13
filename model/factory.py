import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from utils.config_handler import rag_config, chroma_config

api_key = os.getenv("SILICONFLOW_API_KEY")

chat_model: BaseChatModel = ChatOpenAI(
    model=rag_config["chat_model_name"],
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    # 工具调用/Agent 场景用低温采样，避免输出乱码、工具标记泄漏、空响应等不稳定问题
    temperature=0,
    timeout=60,
    max_retries=2,
)

embedding_model: Embeddings = OpenAIEmbeddings(
    model=rag_config["embedding_model_name"],
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    timeout=30,
)

# Rerank 模型配置（Cross-Encoder 精排，复用同一 API Key）
rerank_config = {
    "model": chroma_config["rerank_model"],
    "api_key": api_key,
    "base_url": "https://api.siliconflow.cn/v1",
    "top_n": chroma_config["rerank_top_n"],
}