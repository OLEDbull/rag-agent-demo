import warnings
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools.agent_tools import set_client_ip


def _get_client_ip() -> str:
    """
    从当前 Streamlit 请求中提取真实客户端 IP。
    优先级：X-Forwarded-For 首段 > X-Real-IP > remote_addr。
    生产环境通常由反向代理设置这些头；本地开发会拿到 127.0.0.1。
    内网 IP（127.0.0.1/10.*/192.168.*/172.16-31.*/::1）无法被公网 IP-API 定位，
    返回空字符串让 get_user_location 退化为服务端 IP 自动定位。
    """
    def _is_public_ip(ip: str) -> bool:
        if not ip:
            return False
        if ip in ("127.0.0.1", "::1", "localhost"):
            return False
        if ip.startswith("10.") or ip.startswith("192.168."):
            return False
        # 172.16.0.0 - 172.31.255.255
        if ip.startswith("172."):
            parts = ip.split(".")
            if len(parts) > 1 and 16 <= int(parts[1]) <= 31:
                return False
        return True

    try:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if ctx is None:
            return ""
        session_info = st.runtime.get_instance().get_session_info(ctx.session_id)
        if session_info is None:
            return ""
        headers = getattr(session_info.client, "headers", {}) or {}
        xff = headers.get("X-Forwarded-For", "")
        if xff:
            ip = xff.split(",")[0].strip()
            if _is_public_ip(ip):
                return ip
        xri = headers.get("X-Real-IP", "")
        if xri:
            ip = xri.strip()
            if _is_public_ip(ip):
                return ip
        remote = getattr(session_info.client, "remote_address", "") or ""
        if _is_public_ip(remote):
            return remote
        return ""
    except Exception:
        return ""


st.title("智能助手")
st.divider()

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()
if "message" not in st.session_state:
    st.session_state.message = []

# 侧边栏：开启新会话，清空 Agent 记忆与界面历史
with st.sidebar:
    if st.button("开启新对话"):
        st.session_state.agent.reset_memory()
        st.session_state.message = []
        st.rerun()

for message in st.session_state.message:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state.message.append({"role": "user", "content": prompt})

    # 注入客户端真实 IP 供 get_user_location 工具使用（解决远程部署下定位到服务器位置的问题）
    client_ip = _get_client_ip()
    if client_ip:
        set_client_ip(client_ip)

    # AI 流式回复（历史由 Agent 内部 MemorySaver 按 thread_id 自动管理，无需手动传入）
    with st.chat_message("assistant"):
        res_stream = st.session_state.agent.execute_stream(prompt)
        
        response_messages = []
        
        def stream_response():
            with st.spinner("思考中..."):
                first_chunk = next(res_stream, None)
            
            if first_chunk is not None:
                response_messages.append(first_chunk)
                yield first_chunk
            
            for chunk in res_stream:
                response_messages.append(chunk)
                yield chunk
        
        st.write_stream(stream_response())
        
        full_response = "".join(response_messages)
        st.session_state.message.append({
            "role": "assistant",
            "content": full_response
        })