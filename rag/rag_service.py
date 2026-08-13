"""
总结服务类，用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型回复总结。

混合检索流水线:
  1. 向量召回 (BGE-M3 embedding + Chroma) — 语义相似度
  2. BM25 召回 (rank_bm25) — 关键词精确匹配
  3. RRF 融合 — 按排名倒数加权合并两路结果
  4. Cross-Encoder Rerank (bge-reranker-v2-m3) — 精排，取 Top-N 喂给 LLM
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import VectorStoreService
from rag.bm25_retriever import BM25Retriever
from rag.rag_fusion import rrf_fusion
from rag.rerank_service import RerankService
from utils.prompt_loader import load_rag_prompt
from utils.config_handler import chroma_config
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from utils.logger_handler import logger


def print_prompt(prompt_text: str):
    return prompt_text


class RagSummarizeService(object):
    def __init__(self):
        self.vs = VectorStoreService()
        self.retriever = self.vs.get_retriever()
        self.rerank_service = RerankService()
        self.bm25_retriever = None  # 懒加载：首次查询时从向量库构建 BM25 索引
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chat_model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        return chain

    def _init_bm25(self):
        """从向量库加载所有文档，构建 BM25 索引（懒加载）"""
        all_docs = self.vs.vector_store.get()
        documents = []
        for i, content in enumerate(all_docs.get("documents", [])):
            meta = all_docs["metadatas"][i] if i < len(all_docs.get("metadatas", [])) else {}
            documents.append(Document(page_content=content, metadata=meta))
        if documents:
            self.bm25_retriever = BM25Retriever(documents)
            logger.info(f"BM25 索引懒加载完成，文档数: {len(documents)}")
        else:
            logger.warning("向量库为空，BM25 检索不可用")

    def retriever_docs(self, query: str) -> list[Document]:
        """向量检索召回"""
        return self.retriever.invoke(query)

    def bm25_docs(self, query: str) -> list[Document]:
        """BM25 关键词检索召回"""
        if self.bm25_retriever is None:
            self._init_bm25()
        if self.bm25_retriever is None:
            return []
        return self.bm25_retriever.retrieve(query, top_k=chroma_config["bm25_top_k"])

    def rag_summarize(self, query: str) -> str:
        # === 召回阶段 ===
        # 1. 向量召回（扩大 k，多召回再精筛）
        vector_docs = self.retriever_docs(query)
        logger.info(f"向量召回: {len(vector_docs)} 篇")

        # 2. BM25 关键词召回
        bm25_docs = self.bm25_docs(query)
        logger.info(f"BM25 召回: {len(bm25_docs)} 篇")

        # === 融合阶段 ===
        # 3. RRF 融合（按排名倒数加权合并两路结果）
        if vector_docs or bm25_docs:
            fused_docs = rrf_fusion(vector_docs, bm25_docs, k=chroma_config["rrf_k"])
        else:
            fused_docs = []
            logger.warning("向量与 BM25 召回均为空")

        # === 精排阶段 ===
        # 4. Cross-Encoder Rerank 精排，取 Top-N
        if fused_docs:
            reranked_docs = self.rerank_service.rerank(
                query, fused_docs, top_n=chroma_config["rerank_top_n"]
            )
        else:
            reranked_docs = []

        # === 生成阶段 ===
        # 5. 将精排后的文档作为上下文，交给 LLM 总结
        context = ""
        counter = 0
        for doc in reranked_docs:
            counter += 1
            rerank_score = doc.metadata.get("rerank_score", "N/A")
            context += (f"【参考资料[{counter}】 参考资料：{doc.page_content}"
                        f"|参考元数据：{doc.metadata}"
                        f"|相关性评分：{rerank_score}\n")

        logger.info(f"RAG 流水线完成: 召回 {len(fused_docs)} 篇 → 精排保留 {len(reranked_docs)} 篇")

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("扫地机器人开机自动关机"))
