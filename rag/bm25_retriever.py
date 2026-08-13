"""
BM25 关键词检索器
基于 rank_bm25 库实现，对中文查询做切词后用 BM25 算法评分排序。
比原来的子串匹配更精准：考虑词频(TF)、逆文档频率(IDF)、文档长度归一化。
"""
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from utils.logger_handler import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logger.warning("rank_bm25 未安装，BM25 检索将不可用。请运行: pip install rank_bm25")


def _tokenize_zh(text: str) -> list:
    """
    中文分词：按字符切分（中文无天然空格）。
    BM25 对单字切分效果已足够好，且不依赖 jieba 等外部分词库。
    同时保留英文单词的完整性。
    """
    # 提取英文单词
    tokens = re.findall(r'[a-zA-Z]+', text)
    # 中文按字符切分（过滤标点和空白）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    tokens.extend(chinese_chars)
    return tokens


class BM25Retriever:
    """
    基于 BM25 的关键词检索器。
    在初始化时对所有文档建立 BM25 索引，查询时返回 Top-K 相关文档。
    """

    def __init__(self, documents: list[Document]):
        if BM25Okapi is None:
            raise ImportError("rank_bm25 未安装，请运行: pip install rank_bm25")

        self.documents = documents
        # 对每个文档分词，构建 BM25 索引
        self.corpus_tokens = [_tokenize_zh(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"BM25 索引构建完成，文档数: {len(documents)}")

    def retrieve(self, query: str, top_k: int = 20) -> list[Document]:
        """
        检索与 query 最相关的 top_k 个文档。
        :param query: 查询字符串
        :param top_k: 返回文档数量
        :return: 按相关性降序排列的 Document 列表
        """
        if not self.documents:
            return []

        query_tokens = _tokenize_zh(query)
        if not query_tokens:
            logger.warning(f"查询 [{query}] 分词后为空，BM25 无法检索")
            return []

        # BM25 打分
        scores = self.bm25.get_scores(query_tokens)

        # 按分数降序排序，取 top_k
        scored_docs = list(zip(scores, self.documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 过滤掉得分为 0 的文档（完全无匹配）
        results = [doc for score, doc in scored_docs[:top_k] if score > 0]

        logger.info(f"BM25 检索: query='{query}', tokens={query_tokens}, "
                    f"命中 {len(results)}/{top_k} 篇文档")
        return results
