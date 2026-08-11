import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import requests
import streamlit as st

from config.frontend import (
    ASSISTANT_AVATAR,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    USER_AVATAR,
    WELCOME_MESSAGE,
)

from frontend.styles import load_css

from frontend.api import ChatAPI, ask, ask_stream

from frontend.components import (
    render_header,
    render_history,
    render_sidebar,
    render_sources,
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)

load_css()

def _build_history_payload(messages, exclude_last_user=True):
    """Chuyển session_state.messages thành list history gửi lên backend.

    Chỉ lấy role + content (bỏ sources/latency_ms không cần thiết cho history), loại bỏ tin nhắn user cuối cùng
    (chính là câu hỏi vừa gửi, tránh gửi trùng vì nó đã có trong "query" riêng).
    """
    msgs = messages[:-1] if exclude_last_user else messages
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending" not in st.session_state:
    st.session_state.pending = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

language, streaming, debug = render_sidebar()

render_header()

if len(st.session_state.messages) == 0:
    st.info(WELCOME_MESSAGE)

render_history()

query = st.chat_input("Hỏi về sản phẩm...")

if st.session_state.pending:
    query = st.session_state.pending
    st.session_state.pending = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    history_payload = _build_history_payload(st.session_state.messages)

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        holder = {
            "sources": [],
            "latency_ms": None,
            "error": None,
        }

        answer = ""

        try:
            if streaming:
                with st.spinner("Đang tìm và tổng hợp câu trả lời..."):
                    response = ask_stream(query, language, history=history_payload)

                answer = st.write_stream(ChatAPI.token_generator(response, holder))
            else:
                with st.spinner("Đang tìm và tổng hợp câu trả lời..."):
                    result = ask(query, language, history=history_payload)

                answer = result["answer"]
                holder["sources"] = result["sources"]
                holder["latency_ms"] = result["latency_ms"]

                st.markdown(answer)

        except requests.ConnectionError:
            st.error("Không thể kết nối tới backend!")

        except requests.Timeout:
            st.error("Backend phản hồi quá lâu!")

        except requests.HTTPError as e:
            st.error(str(e))

        except Exception as e:
            st.error(str(e))

        if holder["error"]:
            st.error(holder["error"])

        if answer:
            render_sources(holder["sources"], holder["latency_ms"], debug)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": holder["sources"],
                    "latency_ms": holder["latency_ms"],
                }
            )

            st.rerun()
