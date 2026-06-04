# Utility functions for file handling and visualization

import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import docx
from io import StringIO


# ============================================
# FILE HANDLING FUNCTIONS
# ============================================

def read_text_file(uploaded_file):
    """Read plain text file"""
    content = uploaded_file.read().decode("utf-8")
    return content[:2000]


def read_pdf_file(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text[:2000]


def read_docx_file(uploaded_file):
    """Extract text from Word document"""
    doc = docx.Document(uploaded_file)
    text = " ".join([para.text for para in doc.paragraphs])
    return text[:2000]


def read_file(uploaded_file):
    """Auto-detect file type and extract text"""
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        return read_text_file(uploaded_file)
    elif file_type == "pdf":
        return read_pdf_file(uploaded_file)
    elif file_type == "docx":
        return read_docx_file(uploaded_file)
    else:
        return None


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_ambiguity_chart(results_dict):
    """Create bar chart for ambiguity analysis results"""
    functions = list(results_dict.keys())
    counts = list(results_dict.values())

    colors = ["#2E86AB", "#345995", "#348AA7", "#4CB944",
              "#F4A261", "#E76F51", "#6C91C2", "#8FA6CB",
              "#E74C3C", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(functions, counts, color=colors[:len(functions)])

    ax.set_xlabel("Count", fontsize=11, color="#333333")
    ax.set_title("Semantic Analysis Results", fontsize=14,
                 fontweight="bold", color="#2C3E50", pad=15)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10, color="#333333")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')

    plt.tight_layout()
    return fig


def create_comparison_chart(results_a, results_b, label_a, label_b):
    """Create side-by-side comparison bar chart"""
    functions = list(results_a.keys())
    counts_a = list(results_a.values())
    counts_b = list(results_b.values())

    df = pd.DataFrame({
        'Ambiguity Type': functions,
        label_a: counts_a,
        label_b: counts_b
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(functions))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], counts_a, width,
                   label=label_a, color="#2E86AB")
    bars2 = ax.bar([i + width/2 for i in x], counts_b, width,
                   label=label_b, color="#F4A261")

    ax.set_xticks(x)
    ax.set_xticklabels(functions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Count", fontsize=11, color="#333333")
    ax.set_title("Comparison: Semantic Analysis", fontsize=14,
                 fontweight="bold", color="#2C3E50", pad=15)
    ax.legend(loc='upper right', frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')

    plt.tight_layout()
    return fig


def get_severity_color(count):
    """Return color based on severity"""
    if count >= 6:
        return "#E74C3C"
    elif count >= 3:
        return "#F39C12"
    else:
        return "#27AE60"
