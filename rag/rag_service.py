"""
总结服务类，用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型回复总结
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

def print_prompt(prompt_text:str):
    return prompt_text

class RagSummarizeService(object):
    def __init__(self):
        self.vs = VectorStoreService()
        self.retriever = self.vs.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chat_model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        return chain

    def retriever_docs(self,query:str)->list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self,query:str)->str:

        context_docs = self.retriever_docs(query)

        all_docs = self.vs.vector_store.get()

        keywords = []
        for kw in query.replace("?", "").replace("？", "").split():
            if len(kw) >= 2:
                keywords.append(kw)

        if len(keywords) == 1 and len(keywords[0]) > 4:
            kw = keywords[0]
            for i in range(len(kw) - 1):
                for j in range(i + 2, min(i + 6, len(kw) + 1)):
                    sub = kw[i:j]
                    if len(sub) >= 2:
                        keywords.append(sub)

        exact_matches = []
        for i, doc_content in enumerate(all_docs['documents']):
            if any(kw in doc_content for kw in keywords):
                meta = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                exact_matches.append(Document(page_content=doc_content, metadata=meta))

        if exact_matches:
            combined_docs = exact_matches[:5]
            for doc in context_docs:
                if doc not in combined_docs and len(combined_docs) < 8:
                    combined_docs.append(doc)
        else:
            combined_docs = context_docs

        context = ""
        counter = 0
        for doc in combined_docs:
            counter += 1
            context += f"【参考资料[{counter}】 参考资料：{doc.page_content}|参考元数据：{doc.metadata}\n"
        return self.chain.invoke(
           {
                "input": query,
                "context": context,
           }
        )

if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("扫地机器人开机自动关机"))