import sys
import os
# 定位到项目根目录 D:/AI/agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader,txt_loader, get_file_md5_hex
from utils.file_handler import listdir_with_allowed_type
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name = chroma_config["collection_name"],
            embedding_function = embedding_model,
            persist_directory = chroma_config["persist_directory"]
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chroma_config["chunk_size"],
            chunk_overlap = chroma_config["chunk_overlap"],
            separators = chroma_config["separators"],
            length_function = len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["vector_top_k"]})

    def load_document(self):
        """
        加载文档,转为向量存入向量库
        计算文件的md5做去重
        """
        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_config["md5_hex_store"]), "w",encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_config["md5_hex_store"]), "r",encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a",encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path:str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith(".txt"):
                return txt_loader(read_path)
            else:
                return []

        allowed_files_path = listdir_with_allowed_type(
            chroma_config["data_path"],
            tuple(chroma_config["dataallowed_types"])   
        )

        for file_path in allowed_files_path:
            md5_hex = get_file_md5_hex(file_path)
            if check_md5_hex(md5_hex):
                logger.info(f"文件 {file_path} 已存在向量库中, 跳过加载")
                continue
            try:
                documents:list[Document] = get_file_documents(file_path)

                if not documents:
                    logger.warning(f"加载文档 {file_path} 为空, 跳过加载")
                    continue
                
                split_document:list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.warning(f"加载文档 {file_path} 分片为空, 跳过加载")
                    continue
                
                batch_size = 50
                for i in range(0, len(split_document), batch_size):
                    batch = split_document[i:i + batch_size]
                    self.vector_store.add_documents(batch)
                
                save_md5_hex(md5_hex)

                logger.info(f"加载文档 {file_path} 成功")

            except Exception as e:
                #exc_info=True 会打印异常的详细信息,若为False,则打印异常的类型和信息
                logger.error(f"加载文档 {file_path} 错误信息: {e}",exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("你好")
    for r in res:
        print(r.page_content)
        print("="*20)
