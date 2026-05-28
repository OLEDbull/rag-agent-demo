import os , hashlib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

def get_file_md5_hex(file_path:str):
    """
    获取文件的md5的十六进制字符串
    """
    if not os.path.exists(file_path):
        logger.error(f"md5计算|文件不存在: {file_path}")
        return None
    
    if not os.path.isfile(file_path):
        logger.error(f"md5计算|文件不是普通文件: {file_path}")
        return None
    md5_obj = hashlib.md5()

    chunk_size = 4096  #4KB分片，避免文件过大
    try:
        with open(file_path,"rb") as f:
            while chunk := f.read(chunk_size):     
                md5_obj.update(chunk)
            """
            :=  运算符
            相当于下边四行代码
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"md5计算|文件读取错误: {file_path}, 错误信息: {e}")
        return None
                

def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):
    """
    返回文件夹内的文件列表(允许的文件类型)
    """
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))
    return tuple(files)

def pdf_loader(filepath:str,password:str=None)->list[Document]:
    """
    加载pdf文件
    """
    return PyPDFLoader(filepath,password=password).load()

def txt_loader(filepath:str)->list[Document]:
    """
    加载txt文件
    """
    return TextLoader(filepath, encoding='utf-8').load()
