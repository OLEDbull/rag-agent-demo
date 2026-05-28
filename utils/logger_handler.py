import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.path_tool import get_abs_path
from datetime import datetime



#日志保存根目录
LOG_ROOT = get_abs_path("logs")

#确保目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

#日志格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def get_logger(
    name:str = "agent",
    console_level:logging = logging.INFO,
    file_level:logging = logging.DEBUG,
    log_file:str = None
)->logging.Logger:
    """
    获取日志记录器
    :param name: 日志记录器名称
    :param console_level: 控制台日志级别
    :param file_level: 文件日志级别
    :param log_file: 日志文件名
    :return: 日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    #避免重复添加hander
    if logger.handlers:
        return logger

    #配置控制台hander
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:    
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    return logger

#快捷获取日志记录器
logger = get_logger()
