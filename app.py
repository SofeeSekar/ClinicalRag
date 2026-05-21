import streamlit as st

from modules.document_loader import chunk_text, load_document
from modules.qa import answer_question, compliance_scan, generate_summary
from modules.vector_store import VectorStore

# ─── PAGE CONFIG (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="ClinIQ — Clinical Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ── Base ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #ffffff;
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #f7f9fb;
        border-right: 1px solid #e2e8f0;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

    /* ── Sidebar branding ── */
    .sidebar-brand-name {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1a3c5e;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .sidebar-brand-tag {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 3px;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 0.85rem 0 1rem 0;
    }
    .sidebar-section-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.5rem;
    }

    /* ── Document pill ── */
    .doc-pill-wrap {
        background-color: #e8f0f7;
        border: 1px solid #c5d8ed;
        border-radius: 6px;
        padding: 5px 10px;
        margin: 3px 0;
        font-size: 0.82rem;
        color: #1a3c5e;
        word-break: break-all;
        line-height: 1.4;
    }
    .doc-count {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 0.4rem;
    }

    /* ── Privacy note ── */
    .privacy-note {
        font-size: 0.73rem;
        color: #9ca3af;
        font-style: italic;
        line-height: 1.6;
    }

    /* ── Page header ── */
    .cliniq-header {
        background-color: #1a3c5e;
        padding: 1.25rem 1.75rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .cliniq-header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
    }
    .cliniq-header-sub {
        font-size: 0.85rem;
        color: #a8c4e0;
        margin: 0.2rem 0 0 0;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
    }
    .empty-state-text {
        font-size: 1rem;
        font-weight: 500;
        color: #6b7280;
        margin-top: 1.1rem;
    }
    .empty-state-sub {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.35rem;
    }

    /* ── Chips label ── */
    .chips-label {
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }

    /* ── Chat: user message ── */
    .user-msg-outer {
        display: flex;
        justify-content: flex-end;
        margin: 0.7rem 0;
    }
    .user-bubble {
        background-color: #1a3c5e;
        color: #ffffff;
        border-radius: 16px 16px 4px 16px;
        padding: 0.7rem 1rem;
        max-width: 68%;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    /* ── Chat: AI message ── */
    .ai-msg-outer {
        display: flex;
        justify-content: flex-start;
        margin: 0.7rem 0;
    }
    .ai-bubble {
        background-color: #ffffff;
        color: #111827;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #1a3c5e;
        border-radius: 4px 16px 16px 16px;
        padding: 0.7rem 1rem;
        max-width: 72%;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .sources-block {
        font-size: 0.77rem;
        color: #6b7280;
        border-top: 1px solid #f0f0f0;
        margin-top: 0.55rem;
        padding-top: 0.45rem;
    }

    /* ── Compliance section headers ── */
    .sh-critical {
        font-size: 0.77rem;
        font-weight: 700;
        color: #c53030;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 1.25rem 0 0.5rem 0;
    }
    .sh-review {
        font-size: 0.77rem;
        font-weight: 700;
        color: #b45309;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 1.25rem 0 0.5rem 0;
    }
    .sh-compliant {
        font-size: 0.77rem;
        font-weight: 700;
        color: #276749;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 1.25rem 0 0.5rem 0;
    }

    /* ── Compliance cards ── */
    .card-critical {
        background-color: #fff5f5;
        border-left: 4px solid #e53e3e;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        margin: 0.4rem 0;
    }
    .card-review {
        background-color: #fffbeb;
        border-left: 4px solid #d97706;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        margin: 0.4rem 0;
    }
    .card-compliant {
        background-color: #f0fff4;
        border-left: 4px solid #38a169;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        margin: 0.4rem 0;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.3rem;
        color: #111827;
    }
    .card-excerpt {
        font-size: 0.84rem;
        color: #4b5563;
        font-style: italic;
        margin-bottom: 0.3rem;
    }
    .card-rec {
        font-size: 0.84rem;
        color: #374151;
    }

    /* ── Summary cards ── */
    .summary-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.45rem 0;
        background-color: #fafbfc;
    }
    .summary-label {
        font-size: 0.73rem;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.3rem;
    }
    .summary-value {
        font-size: 0.9rem;
        color: #111827;
        line-height: 1.55;
    }

    /* ── Tab bar: always show all three tabs ── */
    [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: none !important;   /* Firefox */
        -ms-overflow-style: none !important;
        gap: 0 !important;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }

    button[data-baseweb="tab"] {
        flex: 1 1 0% !important;
        min-width: 0 !important;
        white-space: nowrap !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        justify-content: center !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
_defaults: dict = {
    "messages": [],
    "loaded_docs": [],
    "scan_results": {},
    "summary_results": {},
    "chat_input_counter": 0,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "vs" not in st.session_state:
    st.session_state.vs = VectorStore()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown(
        """
        <div style="padding: 1rem 0 0.6rem 0;">
            <div class="sidebar-brand-name">ClinIQ</div>
            <div class="sidebar-brand-tag">Clinical Document Intelligence</div>
        </div>
        <hr class="sidebar-divider" />
        <div class="sidebar-section-label">Load Documents</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="clinical_docs",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Accepted formats: PDF, DOCX",
    )

    # Index newly uploaded files (skip already-loaded ones)
    if uploaded_files:
        existing_names = {d["name"] for d in st.session_state.loaded_docs}
        for f in uploaded_files:
            if f.name not in existing_names:
                with st.spinner(f"Indexing {f.name}..."):
                    pages = load_document(f)
                    chunks = chunk_text(pages)
                    n = st.session_state.vs.add_document(chunks, f.name)
                st.success(f"{f.name}: {n} sections indexed")
                st.session_state.loaded_docs.append({"name": f.name})
                existing_names.add(f.name)

    # Document pills with remove buttons
    if st.session_state.loaded_docs:
        to_remove: list = []
        for i, doc in enumerate(st.session_state.loaded_docs):
            name = doc["name"]
            display = name if len(name) <= 26 else name[:23] + "..."
            col_pill, col_x = st.columns([5, 1])
            with col_pill:
                st.markdown(
                    f'<div class="doc-pill-wrap" title="{name}">{display}</div>',
                    unsafe_allow_html=True,
                )
            with col_x:
                if st.button("x", key=f"rm_{i}", help=f"Remove {name}"):
                    to_remove.append(i)

        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                st.session_state.vs.delete_document(st.session_state.loaded_docs[idx]["name"])
                st.session_state.loaded_docs.pop(idx)
            st.rerun()

        n = len(st.session_state.loaded_docs)
        st.markdown(
            f'<div class="doc-count">{n} document{"s" if n != 1 else ""} loaded</div>',
            unsafe_allow_html=True,
        )

    # Privacy note
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="privacy-note">'
        "All documents are processed locally. No data leaves your device."
        "</div>",
        unsafe_allow_html=True,
    )


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="cliniq-header">
        <div class="cliniq-header-title">ClinIQ</div>
        <div class="cliniq-header-sub">Clinical Document Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Ask ClinIQ", "Compliance Scan", "Document Summary"])


# ── TAB 1: Ask ClinIQ ─────────────────────────────────────────────────────────
with tab1:
    docs = st.session_state.loaded_docs

    if not docs:
        st.markdown(
            """
            <div class="empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"
                     viewBox="0 0 24 24" fill="none" stroke="#d1d5db"
                     stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10 9 9 9 8 9"/>
                </svg>
                <div class="empty-state-text">Load clinical documents to begin</div>
                <div class="empty-state-sub">Upload PDF or DOCX files using the sidebar panel</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Suggested question chips
        st.markdown('<div class="chips-label">Suggested questions</div>', unsafe_allow_html=True)

        SUGGESTED = [
            "What are the contraindications?",
            "What adverse events were reported?",
            "What is the recommended dosage?",
            "What patient populations were excluded?",
        ]
        chip_cols = st.columns(len(SUGGESTED))
        for i, q in enumerate(SUGGESTED):
            with chip_cols[i]:
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("Analysing documents..."):
                        resp = answer_question(q, st.session_state.vs)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": resp["answer"],
                            "sources": resp["sources"],
                        }
                    )
                    st.rerun()

        st.markdown(
            '<hr style="border:none;border-top:1px solid #f0f0f0;margin:0.9rem 0 0.75rem 0;">',
            unsafe_allow_html=True,
        )

        # Chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-msg-outer">'
                    f'<div class="user-bubble">{msg["content"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ai-msg-outer">'
                    f'<div class="ai-bubble">{msg["content"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for s in msg["sources"]:
                            st.markdown(
                                f'<span style="font-size:0.82rem;color:#6b7280;">'
                                f'{s["source"]} &mdash; Page {s["page"]}</span>',
                                unsafe_allow_html=True,
                            )

        # Input row — key rotates on each send to clear the field
        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        input_key = f"chat_input_{st.session_state.chat_input_counter}"
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Question",
                placeholder="Ask a clinical question...",
                label_visibility="collapsed",
                key=input_key,
            )
        with col_btn:
            send = st.button("Send", type="primary", use_container_width=True)

        if send and user_input.strip():
            query = user_input.strip()
            st.session_state.messages.append({"role": "user", "content": query})
            with st.spinner("Analysing documents..."):
                resp = answer_question(query, st.session_state.vs)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": resp["answer"],
                    "sources": resp["sources"],
                }
            )
            st.session_state.chat_input_counter += 1  # rotates key, clears input
            st.rerun()


# ── TAB 2: Compliance Scan ────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<p style="color:#4b5563;font-size:0.9rem;margin-bottom:1.25rem;">'
        "Scan a loaded document against standard clinical safety criteria."
        "</p>",
        unsafe_allow_html=True,
    )

    docs = st.session_state.loaded_docs

    if not docs:
        st.info("Load at least one document using the sidebar to run a compliance scan.")
    else:
        selected_scan = st.selectbox(
            "Select document to scan",
            options=[d["name"] for d in docs],
            key="compliance_select",
        )

        if st.button("Run Compliance Scan", type="primary", key="btn_scan"):
            with st.spinner("Scanning document against clinical criteria..."):
                findings = compliance_scan(selected_scan, st.session_state.vs)
            st.session_state.scan_results[selected_scan] = findings
            st.rerun()

        if selected_scan in st.session_state.scan_results:
            findings_list = st.session_state.scan_results[selected_scan]

            critical = [f for f in findings_list if f.get("level") == "CRITICAL"]
            review   = [f for f in findings_list if f.get("level") == "REVIEW"]
            compliant = [f for f in findings_list if f.get("level") == "COMPLIANT"]

            if critical:
                st.markdown('<div class="sh-critical">Critical Findings</div>', unsafe_allow_html=True)
                for f in critical:
                    st.error(
                        f"**{f['title']}**\n\n"
                        f"*\"{f['excerpt']}\"*\n\n"
                        f"**Recommendation:** {f['recommendation']}"
                    )

            if review:
                st.markdown('<div class="sh-review">Review Required</div>', unsafe_allow_html=True)
                for f in review:
                    st.warning(
                        f"**{f['title']}**\n\n"
                        f"*\"{f['excerpt']}\"*\n\n"
                        f"**Recommendation:** {f['recommendation']}"
                    )

            if compliant:
                st.markdown('<div class="sh-compliant">Compliant</div>', unsafe_allow_html=True)
                for f in compliant:
                    st.success(
                        f"**{f['title']}**\n\n"
                        f"*\"{f['excerpt']}\"*\n\n"
                        f"**Recommendation:** {f['recommendation']}"
                    )


# ── TAB 3: Document Summary ───────────────────────────────────────────────────
with tab3:
    docs = st.session_state.loaded_docs

    if not docs:
        st.info("Load at least one document using the sidebar to generate a summary.")
    else:
        selected_sum = st.selectbox(
            "Select document to summarise",
            options=[d["name"] for d in docs],
            key="summary_select",
        )

        if st.button("Generate Summary", type="primary", key="btn_summary"):
            with st.spinner("Extracting document intelligence..."):
                summary = generate_summary(selected_sum, st.session_state.vs)
            st.session_state.summary_results[selected_sum] = summary
            st.rerun()

        if selected_sum in st.session_state.summary_results:
            sm = st.session_state.summary_results[selected_sum]
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            _scalar_fields = [
                ("document_type", "Document Type"),
                ("purpose", "Purpose"),
            ]
            _list_fields = [
                ("key_entities", "Key Entities"),
                ("critical_dates", "Critical Dates"),
                ("safety_signals", "Safety Signals"),
                ("regulatory_references", "Regulatory References"),
            ]

            for _key, _label in _scalar_fields:
                _val = sm.get(_key, "")
                if _val:
                    st.markdown(
                        f'<div class="summary-card">'
                        f'<div class="summary-label">{_label}</div>'
                        f'<div class="summary-value">{_val}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            for _key, _label in _list_fields:
                _items = sm.get(_key, [])
                if _items:
                    if isinstance(_items, list):
                        _val_html = "<br>".join(f"&bull;&nbsp;{item}" for item in _items)
                    else:
                        _val_html = str(_items)
                    st.markdown(
                        f'<div class="summary-card">'
                        f'<div class="summary-label">{_label}</div>'
                        f'<div class="summary-value">{_val_html}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

