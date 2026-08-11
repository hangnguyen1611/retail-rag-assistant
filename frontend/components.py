import streamlit as st

from config.frontend import (
    APP_NAME,
    ASSISTANT_AVATAR,
    GROUP_LABELS,
    POLICY_LABELS,
    STARTER_QUESTIONS,
    USER_AVATAR,
)


# HEADER
def render_header():
    st.markdown(
        f"""
        <div class="app-badge">✦ Chatbot đa ngôn ngữ </div>
        <div class="main-title">{APP_NAME}</div>""",
        unsafe_allow_html=True,
    )


# SIDEBAR
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""<div class="sidebar-section"> 🛒 {APP_NAME} </div>""",
            unsafe_allow_html=True,
        )

        # New chat
        if st.button("Cuộc trò chuyện mới", icon=":material/add:", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

        # Settings
        st.markdown(
            """<div class="sidebar-section"> ⚙ Cài đặt </div>""",
            unsafe_allow_html=True,
        )

        language = st.selectbox(
            "Ngôn ngữ trả lời", ["auto", "vi", "en"], index=0,
            format_func=lambda v: {"auto": "⌘ Tự động", "vi": "🇻🇳 Tiếng Việt", "en": "🇬🇧 English"}[v],
        )

        col_a, col_b = st.columns(2)
        with col_a:
            streaming = st.toggle("Streaming", value=True)
        with col_b:
            debug = st.toggle("Debug", value=False)

        # Quick suggestions
        st.markdown(
            """
            <div class="sidebar-section"> ⌕ Gợi ý nhanh </div>""",
            unsafe_allow_html=True,
        )

        groups = {}
        for q in STARTER_QUESTIONS:
            groups.setdefault(q["group"], []).append(q)

        for group, items in groups.items():
            st.markdown(
                f"""
                <div class="chip-group-label">
                    {GROUP_LABELS.get(group, group)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            for q in items:
                if st.button(
                    f"{q['icon']}  {q['text']}",
                    use_container_width=True,
                    key=f"starter-{q['text']}",
                ):
                    st.session_state.pending = q["text"]

        # Session info
        st.divider()
        n_turns = len(
            [
                m
                for m in st.session_state.get("messages", [])
                if m["role"] == "user"
            ]
        )
        st.markdown(
            f"""<div class="sidebar-section"> {n_turns} câu hỏi trong phiên này </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Fashion Assistant v2.0 · Colorful UI")

    return language, streaming, debug


# SOURCES
def render_sources(sources, latecy=None, debug=False):
    if not sources:
        return

    with st.expander(f"Nguồn tham khảo ({len(sources)})", expanded=False):
        for s in sources:
            doc_type = s.get("doc_type", "")
            source_name = s.get("title")

            if doc_type == "policy":
                title = POLICY_LABELS.get(source_name, "Chính sách cửa hàng")
                icon = "description"
            else:
                title = (source_name or "Sản phẩm")
                icon = "checkroom"

            st.markdown(
                f"""<div class="sidebar-section"> {icon} {title} </div>""",
                unsafe_allow_html=True,
            )

            if debug and "score" in s:
                st.caption(f"Độ tương đồng: {s['score']:.3f}")

# MESSAGE TOOLBAR
def render_message_toolbar(msg, index):
    with st.container(horizontal=True, gap="small"):
        with st.popover(
            "", icon=":material/content_copy:",
            help="Sao chép câu trả lời",
        ):
            st.code(msg["content"], language=None)

        current = (st.session_state.get("feedback", {}).get(index))
        if st.button(
            "", icon=":material/thumb_up:",
            key=f"up-{index}",
            type=("primary" if current == "up" else "secondary"),
            help="Hữu ích",
        ):
            st.session_state.setdefault("feedback", {})[index] = "up"
            st.toast("Cảm ơn phản hồi của bạn!", icon=":material/thumb_up:")

        current = (st.session_state.get("feedback", {}).get(index))

        if st.button(
            "", icon=":material/thumb_down:",
            key=f"down-{index}",
            type=("primary" if current == "down" else "secondary"),
            help="Chưa hữu ích",
        ):
            st.session_state.setdefault("feedback", {})[index] = "down"
            st.toast("Cảm ơn, mình sẽ cải thiện!", icon=":material/thumb_down:")

        if index == len(st.session_state.messages) - 1:
            if st.button("", icon=":material/refresh:", key=f"regen-{index}", help="Trả lời lại"):
                last_user = None

                for m in reversed(st.session_state.messages[:index]):
                    if m["role"] == "user":
                        last_user = m["content"]
                        break

                if last_user:
                    st.session_state.messages = (st.session_state.messages[:index-1])
                    st.session_state.pending = (last_user)
                    st.rerun()

def render_history():
    if "messages" not in st.session_state:
        return

    for i, msg in enumerate(st.session_state.messages):
        avatar = (
            USER_AVATAR
            if msg["role"] == "user"
            else ASSISTANT_AVATAR
        )

        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Sources
                render_sources(
                    msg.get("sources"),
                    debug=st.session_state.get("debug", False),
                )

                # Latency - nằm ngoài Sources
                if msg.get("latency_ms") is not None:
                    st.markdown(f"""Thời gian phản hồi: {msg['latency_ms']:.0f} ms""")

                # Toolbar
                render_message_toolbar(msg, i)