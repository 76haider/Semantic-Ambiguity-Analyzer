# Streamlit App - Semantic Ambiguity Analyzer for Teachers
# Single Analysis + Compare Mode

import streamlit as st
from analyzer import SemanticAnalyzer, EntityExtractor, get_severity
from papers import PAPERS
from utils import read_file, create_ambiguity_chart, create_comparison_chart, get_severity_color
import pandas as pd
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Semantic Ambiguity Analyzer",
    page_icon="📰",
    layout="wide"
)

# ============================================
# CUSTOM CSS - CLEAN FORMAL COLORS
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #345995;
        margin-top: 1.5rem;
    }
    .result-card {
        background: #FAFAFA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .high-severity {
        border-left: 4px solid #E74C3C;
        background: #FDF2F2;
    }
    .medium-severity {
        border-left: 4px solid #F39C12;
        background: #FEF9F0;
    }
    .low-severity {
        border-left: 4px solid #27AE60;
        background: #F2FDF5;
    }
    .entity-tag {
        display: inline-block;
        background: #2E86AB;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .count-badge {
        display: inline-block;
        background: #345995;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if "history" not in st.session_state:
    st.session_state.history = []

if "analyzer" not in st.session_state:
    st.session_state.analyzer = SemanticAnalyzer()

# ============================================
# HELPER FUNCTIONS
# ============================================
def run_analysis(text, source_name):
    """Run full analysis on text"""
    dynamic_entities = EntityExtractor.extract_dynamic_classes(text)
    spacy_entities = EntityExtractor.extract_spacy_entities(text)
    analysis = st.session_state.analyzer.full_analysis(text)

    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source": source_name,
        "word_count": analysis["word_count"],
        "total_ambiguities": analysis["total"],
        "density": analysis["density"],
        "counts": analysis["counts"],
        "dynamic_entities": dynamic_entities
    })

    return analysis, dynamic_entities, spacy_entities


def display_results(analysis, dynamic_entities, spacy_entities):
    """Display analysis results"""
    counts = analysis["counts"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Words", analysis["word_count"])
    with col2:
        st.metric("⚠️ Total Ambiguities", analysis["total"])
    with col3:
        st.metric("📊 Density (per 100 words)", analysis["density"])
    with col4:
        severity, color = get_severity(analysis["total"])
        st.markdown(f"**Severity:** {severity}")

    st.markdown("---")

    st.markdown('<p class="sub-header">📊 Ambiguity Distribution</p>', unsafe_allow_html=True)
    fig = create_ambiguity_chart(counts)
    st.pyplot(fig)
    st.markdown("---")

    st.markdown('<p class="sub-header">🔍 Detailed Breakdown</p>', unsafe_allow_html=True)

    for func_name, items in analysis["results"].items():
        count = len(items)
        severity_label, color = get_severity(count)

        if count > 0:
            with st.expander(f"{severity_label} {func_name} - {count} found", expanded=(count >= 4)):
                if func_name == "Lexical Ambiguity":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Word:</b> <code>{item['word']}</code><br>
                                <b>Possible Meanings:</b> {', '.join(item['meanings'])}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Referential Ambiguity":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Pronoun:</b> <code>{item['pronoun']}</code><br>
                                <b>Sentence:</b> {item['sentence']}<br>
                                <b>Potential Referents:</b> {item['potential_referents']} nouns in previous sentence
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Scope Ambiguity":
                    for item in items:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Phrase:</b> <code>{item['phrase']}</code><br>
                                <b>Question:</b> {item['ambiguity']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Attachment Ambiguity":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Preposition:</b> <code>{item['preposition']}</code><br>
                                <b>Phrase:</b> {item['phrase']}<br>
                                <b>Sentence:</b> {item['sentence']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Quantifier Ambiguity":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Quantifier:</b> <code>{item['quantifier']}</code><br>
                                <b>Sentence:</b> {item['sentence']}<br>
                                <b>Note:</b> {item['ambiguity']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Temporal Ambiguity":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Vague Time:</b> <code>{item['temporal_word']}</code><br>
                                <b>Sentence:</b> {item['sentence']}<br>
                                <b>Note:</b> {item['ambiguity']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Contextual Ambiguity":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Issue:</b> {item['issue']}<br>
                                <b>Sentence:</b> {item['sentence']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Entity Overlap":
                    for item in items:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Entity:</b> <code>{item['entity']}</code><br>
                                <b>Possible Labels:</b> {', '.join(item['labels'])}<br>
                                <b>Note:</b> {item['ambiguity']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Verb-Object Mismatch":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Verb:</b> <code>{item['verb']}</code> → 
                                <b>Object:</b> <code>{item['object']}</code><br>
                                <b>Sentence:</b> {item['sentence']}<br>
                                <b>Issue:</b> {item['reason']}
                            </div>
                        """, unsafe_allow_html=True)

                elif func_name == "Semantic Coherence":
                    for item in items[:5]:
                        st.markdown(f"""
                            <div class="result-card">
                                <b>Sentence:</b> {item['sentence']}<br>
                                <b>Issue:</b> {item['issue']}
                            </div>
                        """, unsafe_allow_html=True)

    # Entity classes
    st.markdown("---")
    st.markdown('<p class="sub-header">🏷️ Detected Entity Classes (Dynamic)</p>', unsafe_allow_html=True)

    if dynamic_entities:
        cols = st.columns(min(len(dynamic_entities), 4))
        for i, (class_name, count) in enumerate(dynamic_entities.items()):
            with cols[i % len(cols)]:
                st.markdown(f"""
                    <div style="background:#EEF5F9; padding:1rem; border-radius:8px; text-align:center;">
                        <h4>{class_name}</h4>
                        <h2 style="color:#2E86AB;">{count}</h2>
                        <small>keywords matched</small>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No specific entity classes detected. Text may be too short or generic.")

    if spacy_entities:
        st.markdown('<p class="sub-header">📍 Named Entities (spaCy)</p>', unsafe_allow_html=True)
        for entity_type, entities in spacy_entities.items():
            tags = " ".join([f'<span class="entity-tag">{e}</span>' for e in entities[:8]])
            st.markdown(f"**{entity_type}:** {tags}", unsafe_allow_html=True)


# ============================================
# MAIN APP
# ============================================
st.markdown('<h1 class="main-header">📰 Semantic Ambiguity Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#666; font-size:1.1rem;">NLP Project - Semantic Ambiguity Detection</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#2E86AB; font-size:0.95rem;"><b>Teacher:</b> Sir Muhammad Ali Samo | <b>Course:</b> Natural Language Processing</p>', unsafe_allow_html=True)

mode = st.radio(
    "Select Mode:",
    ["📋 Single Analysis", "🔄 Compare Mode"],
    horizontal=True
)

st.markdown("---")

# ============================================
# SINGLE ANALYSIS MODE
# ============================================
if mode == "📋 Single Analysis":
    st.markdown('<p class="sub-header">📥 Select Input Source</p>', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method:",
        ["📚 Pre-built Paper", "✏️ Manual Text", "📁 Upload File"],
        horizontal=True
    )

    text_to_analyze = ""
    source_name = ""

    if input_method == "📚 Pre-built Paper":
        paper_choice = st.selectbox("Select a newspaper:", list(PAPERS.keys()))
        text_to_analyze = PAPERS[paper_choice]
        source_name = paper_choice

        with st.expander("📄 Preview Text"):
            st.text_area("Preview", text_to_analyze, height=200, disabled=True, label_visibility="collapsed")

    elif input_method == "✏️ Manual Text":
        text_to_analyze = st.text_area(
            "Enter text (250-300 words recommended):",
            height=200,
            placeholder="Paste your newspaper text here..."
        )
        source_name = "Manual Input"

    elif input_method == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "Upload .txt, .pdf, or .docx file:",
            type=["txt", "pdf", "docx"]
        )
        if uploaded_file:
            text_to_analyze = read_file(uploaded_file)
            source_name = f"File: {uploaded_file.name}"

            with st.expander("📄 Extracted Text"):
                st.text_area("Extracted", text_to_analyze, height=200, disabled=True, label_visibility="collapsed")

    if st.button("🔍 Analyze Ambiguity", type="primary", use_container_width=True):
        if text_to_analyze.strip():
            with st.spinner("Analyzing semantic ambiguity..."):
                analysis, dynamic_entities, spacy_entities = run_analysis(text_to_analyze, source_name)
                display_results(analysis, dynamic_entities, spacy_entities)
        else:
            st.warning("⚠️ Please provide text to analyze.")

# ============================================
# COMPARE MODE
# ============================================
else:
    st.markdown('<p class="sub-header">🔄 Compare Two Texts</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Text A**")
        input_a = st.radio("Source A:", ["Pre-built", "Manual", "Upload"],
                           key="a", horizontal=True)

        text_a = ""
        label_a = ""

        if input_a == "Pre-built":
            paper_a = st.selectbox("Select paper:", list(PAPERS.keys()), key="pa")
            text_a = PAPERS[paper_a]
            label_a = paper_a
            with st.expander("Preview A"):
                st.text(text_a[:300] + "...")

        elif input_a == "Manual":
            text_a = st.text_area("Enter Text A:", height=150, key="ta")
            label_a = "Manual Input A"

        elif input_a == "Upload":
            file_a = st.file_uploader("Upload File A:", type=["txt", "pdf", "docx"], key="fa")
            if file_a:
                text_a = read_file(file_a)
                label_a = f"File: {file_a.name}"

    with col2:
        st.markdown("**Text B**")
        input_b = st.radio("Source B:", ["Pre-built", "Manual", "Upload"],
                           key="b", horizontal=True)

        text_b = ""
        label_b = ""

        if input_b == "Pre-built":
            paper_b = st.selectbox("Select paper:", list(PAPERS.keys()), key="pb")
            text_b = PAPERS[paper_b]
            label_b = paper_b
            with st.expander("Preview B"):
                st.text(text_b[:300] + "...")

        elif input_b == "Manual":
            text_b = st.text_area("Enter Text B:", height=150, key="tb")
            label_b = "Manual Input B"

        elif input_b == "Upload":
            file_b = st.file_uploader("Upload File B:", type=["txt", "pdf", "docx"], key="fb")
            if file_b:
                text_b = read_file(file_b)
                label_b = f"File: {file_b.name}"

    if st.button("🔄 Compare Ambiguity", type="primary", use_container_width=True):
        if text_a.strip() and text_b.strip():
            with st.spinner("Analyzing both texts..."):
                analysis_a, dynamic_a, spacy_a = run_analysis(text_a, label_a)
                analysis_b, dynamic_b, spacy_b = run_analysis(text_b, label_b)

                st.markdown("---")
                st.markdown('<p class="sub-header">📊 Comparison Summary</p>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Words (A vs B)", f"{analysis_a['word_count']} / {analysis_b['word_count']}")
                with c2:
                    st.metric("Total Ambiguities", f"{analysis_a['total']} / {analysis_b['total']}")
                with c3:
                    st.metric("Density", f"{analysis_a['density']} / {analysis_b['density']}")
                with c4:
                    winner = "Text A" if analysis_a['total'] > analysis_b['total'] else "Text B"
                    st.metric("More Ambiguous", winner)

                fig = create_comparison_chart(
                    analysis_a["counts"], analysis_b["counts"],
                    label_a[:30], label_b[:30]
                )
                st.pyplot(fig)

                st.markdown("---")
                st.markdown('<p class="sub-header">🔍 Side-by-Side Results</p>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(f"**{label_a[:40]}**")
                    st.markdown(f"Total: {analysis_a['total']} | Words: {analysis_a['word_count']}")
                    for func, count in analysis_a["counts"].items():
                        sev, col = get_severity(count)
                        st.markdown(f"{sev} {func}: **{count}**")

                with col_b:
                    st.markdown(f"**{label_b[:40]}**")
                    st.markdown(f"Total: {analysis_b['total']} | Words: {analysis_b['word_count']}")
                    for func, count in analysis_b["counts"].items():
                        sev, col = get_severity(count)
                        st.markdown(f"{sev} {func}: **{count}**")
        else:
            st.warning("⚠️ Please provide both texts to compare.")

# ============================================
# SIDEBAR - HISTORY
# ============================================
with st.sidebar:
    st.markdown("## 📜 Analysis History")
    st.markdown("---")

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            sev, color = get_severity(entry["total_ambiguities"])

            st.markdown(f"""
                <div style="background:#FAFAFA; padding:0.8rem; border-radius:8px; 
                            border-left:4px solid {color}; margin-bottom:0.5rem;">
                    <small>{entry['timestamp']}</small><br>
                    <b>{entry['source'][:35]}</b><br>
                    Words: {entry['word_count']} | Ambiguities: {entry['total_ambiguities']}<br>
                    {sev} | Density: {entry['density']}
                </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No analysis yet. Start analyzing!")

# ============================================
# FOOTER
# ============================================
st.markdown(
    '<p style="text-align:center; color:#999; font-size:0.85rem;">'
    'NLP Project - Semantic Ambiguity Analyzer | Built with Streamlit & spaCy<br>'
    '<b>Muhammad Haider (CSC-23S-061)</b> | Taha Shabbir (CSC-23S-062) | Ali Baba (CSC-23S-093)</p>',
    unsafe_allow_html=True
)
