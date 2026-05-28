import warnings
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
import streamlit as st
from agent.react_agent import ReactAgent

st.title("智能助手")
st.divider()

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()
if "message" not in st.session_state:
    st.session_state.message = []

for message in st.session_state.message:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state.message.append({"role": "user", "content": prompt})

    # AI 流式回复
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