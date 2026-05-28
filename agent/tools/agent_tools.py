import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.file_handler import get_file_md5_hex



rag_service = RagSummarizeService()
external_data = {}
user_ids = ["user1", "user2", "user3", "user4", "user5"]
month_arr = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]

@tool(description="根据查询问题，从知识库中提取相关文档并生成摘要")
def rag_summarize(query:str)->str:
    
    return rag_service.rag_summarize(query)

@tool(description="根据城市名称，获取该城市的天气，以消息字符串形式返回")
def get_weather(city:str)->str:
    return f"{city}的天气是晴朗的"
   
@tool(description="获取用户当前所在城市，以纯字符串形式返回")
def get_user_location()->str:
    return random.choice(["北京","上海","广州","深圳"])
    
@tool(description="获取用户ID，以纯字符串形式返回")
def get_user_id()->str:
    return random.choice(user_ids)

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month()->str:
    return random.choice(month_arr)

@tool(description="为报告填充上下文信息")
def fill_context_for_report()->str:
    return "报告上下文信息已填充"

def generate_external_data()->str:
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")
        
        with open(external_data_path,"r",encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr:list[str] = line.strip().split(",")
                user_id:str = arr[0].replace('"',"")
                feature:str = arr[1].replace('"',"")
                efficiency:str = arr[2].replace('"',"")
                consumables:str = arr[3].replace('"',"")
                comparison:str = arr[4].replace('"',"")
                time:str = arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征":feature,
                    "效率":efficiency,
                    "耗材":consumables,
                    "对比":comparison
                }
                

@tool(description="根据用户ID和月份，获取该用户在该月份的外部数据，以消息字符串形式返回")
def fetch_external_data(user_id:str,month:str)->str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.error(f"未能检索到用户ID {user_id}在{month}的外部数据")
        return ""


    