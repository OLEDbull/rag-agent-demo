"""
RRF (Reciprocal Rank Fusion) 融合算法
将多路检索结果按排名倒数加权融合，无需对齐分数尺度。
公式: rrf_score(d) = Σ 1 / (k + rank_i(d))
  - k: 平滑常数（通常取 60），值越大对低排名文档越友好
  - rank_i(d): 文档 d 在第 i 路检索结果中的排名（从 0 开始）
"""
from langchain_core.documents import Document
from utils.logger_handler import logger


def rrf_fusion(vector_docs: list[Document],
               bm25_docs: list[Document],
               k: int = 60) -> list[Document]:
    """
    对向量检索和 BM25 检索的结果做 RRF 融合。
    :param vector_docs: 向量检索结果（按相似度降序）
    :param bm25_docs: BM25 检索结果（按 BM25 分数降序）
    :param k: RRF 平滑常数
    :return: 融合后按 RRF 分数降序排列的文档列表（去重）
    """
    # 用文档内容作为去重 key（不同检索可能召回同一文档）
    doc_scores = {}  # key: page_content, value: {"score": float, "doc": Document}

    # 向量检索结果：按排名累加 RRF 分数
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        score = 1.0 / (k + rank)
        if key in doc_scores:
            doc_scores[key]["score"] += score
        else:
            doc_scores[key] = {"score": score, "doc": doc}

    # BM25 检索结果：按排名累加 RRF 分数
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        score = 1.0 / (k + rank)
        if key in doc_scores:
            doc_scores[key]["score"] += score
        else:
            doc_scores[key] = {"score": score, "doc": doc}

    # 按 RRF 总分降序排列
    fused = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    result = [item["doc"] for item in fused]

    logger.info(f"RRF 融合: 向量={len(vector_docs)} 篇 + BM25={len(bm25_docs)} 篇 "
                f"→ 融合去重后 {len(result)} 篇 (k={k})")
    return result
