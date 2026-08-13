"""
Cross-Encoder Rerank 服务
调用 SiliconFlow 的 rerank API（bge-reranker-v2-m3）对召回文档做精排。

与 Embedding（双塔模型，query 和 doc 独立编码）不同，
Cross-Encoder 将 query 和 doc 拼接后一起送入模型，能捕捉更细粒度的交互特征，
在精排阶段显著提升相关性排序质量。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from langchain_core.documents import Document
from model.factory import rerank_config
from utils.logger_handler import logger


class RerankService:
    """调用 SiliconFlow Cross-Encoder 对文档做精排"""

    def __init__(self):
        self.model = rerank_config["model"]
        self.api_key = rerank_config["api_key"]
        self.base_url = rerank_config["base_url"]
        self.top_n = rerank_config["top_n"]

    def rerank(self, query: str, documents: list[Document],
               top_n: int = None) -> list[Document]:
        """
        对 documents 按 query 相关性做精排。
        :param query: 用户查询
        :param documents: 召回阶段得到的候选文档
        :param top_n: 最终保留的文档数，None 则用默认配置
        :return: 按相关性降序排列的 Document 列表
        """
        if not documents:
            return []

        if not self.api_key:
            logger.warning("Rerank API Key 未配置，跳过精排，直接返回原始文档")
            return documents[:top_n or self.top_n]

        top_n = top_n or self.top_n
        # API 要求 top_n 不超过文档数
        top_n = min(top_n, len(documents))

        # 准备请求数据
        doc_texts = [doc.page_content for doc in documents]
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": top_n,
            "return_documents": False,  # 只需要 index 和 score，文档内容已有
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # 解析结果：results 按 relevance_score 降序排列
            results = data.get("results", [])
            if not results:
                logger.warning("Rerank API 返回空结果，返回原始文档前 top_n 条")
                return documents[:top_n]

            # 按 API 返回的 index 映射回原始 Document
            reranked_docs = []
            for r in results:
                idx = r.get("index")
                score = r.get("relevance_score", 0)
                if idx is not None and 0 <= idx < len(documents):
                    doc = documents[idx]
                    # 将 rerank 分数写入 metadata，方便后续调试/日志
                    doc.metadata["rerank_score"] = round(score, 4)
                    reranked_docs.append(doc)

            logger.info(f"Rerank 精排完成: {len(documents)} → {len(reranked_docs)} 篇, "
                        f"模型={self.model}, top_score={results[0].get('relevance_score', 'N/A'):.4f}")
            return reranked_docs

        except requests.exceptions.RequestException as e:
            logger.error(f"Rerank API 请求失败: {e}")
            # 降级：返回原始文档前 top_n 条
            return documents[:top_n]
        except (KeyError, ValueError) as e:
            logger.error(f"Rerank API 响应解析失败: {e}")
            return documents[:top_n]
