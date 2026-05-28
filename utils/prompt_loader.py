import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompt():
    """
    加载系统提示词
    """
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except Exception as e:
        logger.error(f"加载系统提示词|错误信息: {e}")
        raise e
    try:
        return open(system_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析系统提示词|错误信息: {e}")
        raise e

def load_rag_prompt():
    """
    加载rag_summarize提示词
    """
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except Exception as e:
        logger.error(f"加载系统提示词|错误信息: {e}")
        raise e
    try:
        return open(rag_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析rag_summarize提示词|错误信息: {e}")
        raise e

def load_report_prompt():
    """
    加载report_prompt提示词
    """
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except Exception as e:
        logger.error(f"加载report_prompt提示词|错误信息: {e}")
        raise e
    try:
        return open(report_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析report_prompt提示词|错误信息: {e}")  
        raise e