import sys
import os
import json
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from utils.logger_handler import logger
from agent.tools.agent_tools import (rag_summarize, fetch_external_data, fill_context_for_report,
                                     get_user_id, get_current_month, get_user_location, get_weather)


class ReactAgent:
    def __init__(self):
        system_prompt = load_system_prompt()
        self.agent = create_agent(
            model=chat_model,
            tools=[rag_summarize, fetch_external_data, fill_context_for_report, get_user_id,
                   get_current_month, get_user_location, get_weather],
            system_prompt=system_prompt,
        )

    def _clean_content(self, content: str) -> str:
        if not content or not isinstance(content, str):
            return ""

        # 剥离模型偶发泄漏的工具调用标记及其包裹内容
        content = re.sub(r'TECHNOAI_TOOLS_STARTMARKER.*?TECHNOAI_TOOLS_ENDMARKER', '', content, flags=re.DOTALL)
        content = re.sub(r'TECHNOAI_TOOLS_(?:START|END)MARKER', '', content)

        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                return ""
            except (json.JSONDecodeError, TypeError):
                pass

        prev = None
        while prev != content:
            prev = content
            content = re.sub(r'\{[^{}]*\}', '', content)

        content = re.sub(r'[\{\}]', '', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        content = content.strip()
        content = content.strip('"\'')

        if len(content) < 2:
            return ""

        return content

    def execute_stream(self, query: str, history: list = None):
        try:
            yield from self._execute_stream_impl(query, history)
        except Exception as e:
            import traceback
            error_msg = "处理请求时出错: " + str(e) + "\n" + traceback.format_exc()
            for char in error_msg:
                yield char

    def _stream_agent(self, input_dict: dict, show_tool_indicator: bool = True):
        """
        执行一次 Agent 流式调用。
        :param input_dict: Agent 输入（含 messages）
        :param show_tool_indicator: 是否向前端 yield 工具调用提示
        :return: (清洗后的最终回答, get_weather 的真实返回文本)
        """
        seen_ids = set()
        final_answer = ""
        weather_fact = None  # 捕获 get_weather 的真实返回，用于反编造兜底

        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            for msg in chunk.get("messages", []):
                msg_id = id(msg)
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)

                # 捕获 get_weather 工具的真实返回结果（模型偶发会忽略它自己编数字）
                if getattr(msg, 'type', None) == "tool" and getattr(msg, 'name', None) == "get_weather":
                    weather_fact = msg.content

                if not hasattr(msg, 'type') or msg.type != "ai":
                    continue

                # 先处理工具调用：工具调用消息的 content 常常为空，不能因 content 为空而被跳过
                if getattr(msg, 'tool_calls', None):
                    if show_tool_indicator:
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', 'unknown')
                            yield f"\n\n🔧 正在调用工具: {tool_name}...\n\n"
                    continue

                if not hasattr(msg, 'content') or not msg.content:
                    continue

                content = msg.content
                if not isinstance(content, str) or not content.strip():
                    continue

                cleaned = self._clean_content(content)
                if cleaned:
                    final_answer = cleaned

        return final_answer, weather_fact

    def _execute_stream_impl(self, query: str, history: list = None):
        # 组装多轮对话历史，让 Agent 具备记忆；仅保留最近的若干轮，避免 token 膨胀
        messages = []
        for m in (history or [])[-20:]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        input_dict = {"messages": messages}

        # 首次执行（展示工具调用过程）
        final_answer, weather_fact = yield from self._stream_agent(input_dict, show_tool_indicator=True)

        # 模型偶发中途空响应/链条截断时，自动重试一次（不重复展示工具提示）
        if not final_answer:
            logger.warning("Agent 首次响应为空，自动重试一次")
            final_answer, weather_fact = yield from self._stream_agent(input_dict, show_tool_indicator=False)

        # 反编造兜底：模型最终答案里的温度若与 get_weather 真实返回值偏差 > 3℃，
        # 判定为编造，强制把真实天气数据顶到回答最前面，确保用户看到的是工具真实数据。
        if weather_fact and final_answer:
            # 统一温度符号：模型可能输出半角 "°C"，工具返回全角 "℃"，先归一化再比较
            norm_fact = weather_fact.replace("°C", "℃")
            norm_ans = final_answer.replace("°C", "℃")
            real_temp = re.search(r'(\d+(?:\.\d+)?)℃', norm_fact)
            if real_temp:
                real_val = float(real_temp.group(1))
                ans_temp = re.search(r'(\d+(?:\.\d+)?)℃', norm_ans)
                if ans_temp:
                    ans_val = float(ans_temp.group(1))
                    if abs(ans_val - real_val) > 3.0:
                        logger.warning(f"检测到天气温度编造（模型 {ans_val}℃ vs 真实 {real_val}℃），强制注入真实数据")
                        final_answer = f"🌤️ 实时天气（工具获取）：{weather_fact}\n\n" + final_answer

        if final_answer:
            for char in final_answer:
                yield char
        else:
            fallback = "抱歉，我暂时无法回答这个问题，请稍后再试。"
            for char in fallback:
                yield char


if __name__ == "__main__":
    react_agent = ReactAgent()
    for chunk in react_agent.execute_stream("扫地机器人怎么保养？"):
        try:
            print(chunk, end="", flush=True)
        except UnicodeEncodeError:
            pass