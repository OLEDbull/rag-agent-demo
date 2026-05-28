import sys
import os
import json
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
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

    def execute_stream(self, query: str):
        try:
            yield from self._execute_stream_impl(query)
        except Exception as e:
            import traceback
            error_msg = "处理请求时出错: " + str(e) + "\n" + traceback.format_exc()
            for char in error_msg:
                yield char

    def _execute_stream_impl(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        seen_ids = set()
        final_answer = ""

        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            messages = chunk.get("messages", [])

            for msg in messages:
                msg_id = id(msg)
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)

                if not hasattr(msg, 'type') or msg.type != "ai":
                    continue
                if not hasattr(msg, 'content') or not msg.content:
                    continue

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        yield f"\n\n🔧 正在调用工具: {tool_name}...\n\n"
                    continue

                content = msg.content
                if not isinstance(content, str) or not content.strip():
                    continue

                cleaned = self._clean_content(content)
                if cleaned:
                    final_answer = cleaned

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