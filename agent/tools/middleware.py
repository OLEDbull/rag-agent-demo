import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langgraph.types import Command, AgentState, Runtime, ModelRequest
from utils.prompt_loader import load_system_prompt, load_report_prompt
from utils.logger_handler import logger

def monitor_tool(state: AgentState):
    """
    监控工具调用
    """
    if 'messages' in state:
        for message in state['messages']:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    logger.info(f"[monitor_tool] 调用工具: {tool_call.get('name', 'unknown')}")
                    logger.info(f"[monitor_tool] 调用参数: {tool_call.get('args', {})}")
    return state

def log_before_model(state: AgentState):
    """
    在模型调用前记录日志
    """
    logger.info(f"[log_before_model]即将执行模型调用，带有{len(state)} 个状态记录")
    if 'messages' in state:
        logger.debug(f"[log_before_model]{type(state['messages'])}")
    return state

def report_prompt_switch(request: ModelRequest):
    """
    根据上下文切换提示词
    """
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()
    return load_system_prompt()