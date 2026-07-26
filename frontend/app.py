import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
API_URL = f"{BACKEND_URL}/chat"

st.title("Retail Assistant - Thời trang")

with st.sidebar:
    language = st.selectbox("Ngôn ngữ / Language", ["auto", "vi", "en"], index=0)
    st.caption("Backend: " + API_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Nguồn tham khảo ({msg['latency_ms']:.0f} ms)"):
                for s in msg["sources"]:
                    st.write(f"- `{s['doc_type']}` **{s['doc_id']}** (score: {s['score']:.3f})")

query = st.chat_input("Hỏi về sản phẩm, đổi trả, vận chuyển...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm câu trả lời..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"query": query, "language": language},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                sources = data.get("sources", [])
                latency_ms = data.get("latency_ms", 0)

                st.write(answer)
                if sources:
                    with st.expander(f"Nguồn tham khảo ({latency_ms:.0f} ms)"):
                        for s in sources:
                            st.write(f"- `{s['doc_type']}` **{s['doc_id']}** (score: {s['score']:.3f})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency_ms": latency_ms,
                })

            except requests.exceptions.ConnectionError:
                error_msg = "Không kết nối được tới backend. Kiểm tra `uvicorn api.main:app --reload` đang chạy chưa."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.Timeout:
                error_msg = "Backend phản hồi quá lâu (timeout 30s). Thử lại sau."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.HTTPError as e:
                error_msg = f"Backend trả lỗi: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})