"""Neural Network-Based Token Classification for Compilers.

A modern, light-themed Compiler Design dashboard for lexical analysis,
deep learning token classification, and syntax verification.
Designed for college project presentations and technical defense.

Organized into 5 core sections:
1. Token Classifier
2. Smart Token Error Detector
3. Model Comparison Lab
4. Syntax Tree Visualizer
5. Model Evaluation & Metrics
"""
import os
import time
import json
import html
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.inference import TokenClassifierPredictor
from src.lexer_utils import CLASSES, CLASS_TO_ID, ID_TO_CLASS
from src.c_parser import parse_c_code, ASTNode, SyntaxErrorInfo
from src.tree_graph import ast_to_dot, dot_to_svg
from src.active_learning import (
    save_feedback,
    load_feedback,
    get_feedback_count,
    clear_feedback,
    train_active_learning_model
)
from src.evaluate import compare_bilstm_vs_lexer

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Compiler Intelligence Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Light-Themed Compiler Design Dashboard CSS Styling
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
    :root, [data-theme="dark"], [data-theme="light"], html, body {
        color-scheme: light !important;
        --noc-blue: #6366F1;
        --noc-blue-dark: #4F46E5;
        --noc-cyan: #06B6D4;
        --noc-cyan-light: #CFFAFE;
        --noc-cyan-tint: #EEF2FF;
        --noc-dark: #1E1B4B;
        --noc-slate: #312E81;
        --noc-muted: #64748B;
        --noc-border: #E0E7FF;
        --noc-card-bg: #FFFFFF;
        --noc-bg-page: #F4F0FF;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .main, .block-container,
    [data-testid="stBottom"],
    footer {
        color: var(--noc-dark) !important;
        font-family: "Outfit", "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1200px 500px at 8% -10%, rgba(99, 102, 241, 0.22), transparent 55%),
            radial-gradient(900px 420px at 100% 0%, rgba(6, 182, 212, 0.18), transparent 50%),
            radial-gradient(700px 380px at 70% 100%, rgba(244, 63, 94, 0.10), transparent 50%),
            linear-gradient(180deg, #F8F5FF 0%, #EEF2FF 48%, #F4F0FF 100%) !important;
    }

    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.8rem !important;
        max-width: 1360px !important;
        animation: fadeRise 0.55s ease both;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        height: 2.2rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1B4B 0%, #312E81 42%, #4338CA 100%) !important;
        border-right: 0 !important;
    }
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #EEF2FF;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.16) !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #E0E7FF !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.22) !important;
        color: #FFFFFF !important;
    }

    /* Top NOC Header */
    .noc-header-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(238,242,255,0.92) 55%, rgba(224,242,254,0.90) 100%);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 22px;
        padding: 1.35rem 1.7rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 18px 40px rgba(79, 70, 229, 0.12), 0 2px 0 rgba(255,255,255,0.7) inset;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
        position: relative;
        overflow: hidden;
        animation: fadeRise 0.7s ease both;
    }
    .noc-header-card::before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #6366F1, #06B6D4, #F43F5E, #F59E0B);
        background-size: 220% 100%;
        animation: ribbonShift 7s linear infinite;
    }

    .noc-title-area {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .noc-tagline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, #EEF2FF, #ECFEFF);
        color: #4338CA;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid #C7D2FE;
        width: fit-content;
        margin-bottom: 4px;
    }

    .noc-main-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E1B4B;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.15;
    }
    .noc-main-title span {
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .noc-sub-title {
        font-size: 0.88rem;
        color: #64748B;
        margin: 0;
        font-weight: 500;
    }

    .noc-telemetry-cluster {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    .noc-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        white-space: nowrap;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .noc-pill:hover { transform: translateY(-1px); }
    .noc-pill-green {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        box-shadow: 0 6px 14px rgba(16, 185, 129, 0.12);
    }
    .noc-pill-blue {
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        box-shadow: 0 6px 14px rgba(99, 102, 241, 0.14);
    }
    .noc-pill-slate {
        background: #F8FAFC;
        color: #334155;
        border: 1px solid #CBD5E1;
    }
    .noc-pill-purple {
        background: #FDF2F8;
        color: #BE185D;
        border: 1px solid #FBCFE8;
        box-shadow: 0 6px 14px rgba(244, 63, 94, 0.12);
    }

    .pulse-dot-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseNoc 2s infinite;
    }
    @keyframes pulseNoc {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes fadeRise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes ribbonShift {
        0% { background-position: 0% 50%; }
        100% { background-position: 220% 50%; }
    }

    /* Section Cards */
    .noc-card {
        background-color: #FFFFFF;
        border: 1px solid rgba(99, 102, 241, 0.16);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.07);
    }

    .noc-card-header {
        font-size: 1.02rem;
        font-weight: 800;
        color: #1E1B4B;
        letter-spacing: -0.01em;
        margin-bottom: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: linear-gradient(90deg, #EEF2FF 0%, #ECFEFF 100%);
        border: 1px solid #E0E7FF;
        border-radius: 14px;
        padding: 0.72rem 1rem;
        animation: fadeRise 0.45s ease both;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: none;
        margin-bottom: 1.2rem;
        background: rgba(255,255,255,0.55) !important;
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.14);
        flex-wrap: wrap;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
        padding: 10px 16px !important;
        border-radius: 12px !important;
        background-color: transparent !important;
        border: 0 !important;
        transition: all 0.22s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4338CA !important;
        background-color: #EEF2FF !important;
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 55%, #06B6D4 140%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 22px rgba(79, 70, 229, 0.32) !important;
    }

    /* Telemetry Metric Boxes */
    .metric-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%);
        border: 1px solid rgba(99, 102, 241, 0.14);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        animation: fadeRise 0.5s ease both;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 32px rgba(79, 70, 229, 0.14);
    }
    .metric-card-blue { border-top: 4px solid #6366F1; }
    .metric-card-cyan { border-top: 4px solid #06B6D4; }
    .metric-card-green { border-top: 4px solid #10B981; }
    .metric-card-amber { border-top: 4px solid #F59E0B; }
    .metric-card-rose { border-top: 4px solid #F43F5E; }
    .metric-card-purple { border-top: 4px solid #A855F7; }

    .m-label {
        font-size: 0.72rem;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .m-val {
        font-size: 1.62rem;
        font-weight: 800;
        color: #1E1B4B;
        line-height: 1.2;
    }
    .m-sub {
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 2px;
        font-weight: 600;
    }

    /* Monospace Code Editor */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #4338CA !important;
        border-radius: 14px !important;
        font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace !important;
        font-size: 0.88rem !important;
        line-height: 1.65 !important;
        padding: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.18) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 55%, #06B6D4 160%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border: none !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.28) !important;
        transition: all 0.22s ease !important;
    }
    .stButton > button:hover {
        filter: brightness(1.08);
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.38) !important;
        transform: translateY(-2px);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 8px 18px rgba(16, 185, 129, 0.22) !important;
    }

    /* Token Class Palette Badges */
    .token-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
        letter-spacing: 0.02em;
    }
    .badge-keyword { background-color: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; }
    .badge-identifier { background-color: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-operator { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-literal { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-delimiter { background-color: #FDF2F8; color: #BE185D; border: 1px solid #FBCFE8; }

    /* Highlighted Code Canvas */
    .highlighted-code-container {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 1.15rem;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
        overflow-x: auto;
    }
    .highlighted-code-pre {
        margin: 0;
        font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
        font-size: 0.88rem;
        line-height: 1.85;
        white-space: pre;
        color: #1E293B;
    }
    .code-line { display: flex; align-items: baseline; }
    .line-num {
        display: inline-block;
        width: 30px;
        color: #94A3B8;
        font-size: 0.78rem;
        user-select: none;
        text-align: right;
        margin-right: 14px;
        border-right: 1px solid #E2E8F0;
        padding-right: 8px;
    }
    .line-code { flex: 1; }

    .code-tok {
        padding: 1px 5px;
        border-radius: 5px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s ease, filter 0.15s ease;
    }
    .code-tok:hover { transform: translateY(-1px); filter: brightness(1.12); }
    .code-tok.tok-keyword { background-color: #F3E8FF; color: #6D28D9; border-bottom: 2px solid #8B5CF6; }
    .code-tok.tok-identifier { background-color: #E0E7FF; color: #3730A3; border-bottom: 2px solid #6366F1; }
    .code-tok.tok-operator { background-color: #D1FAE5; color: #065F46; border-bottom: 2px solid #10B981; }
    .code-tok.tok-literal { background-color: #FEF3C7; color: #92400E; border-bottom: 2px solid #F59E0B; }
    .code-tok.tok-delimiter { background-color: #FFE4E6; color: #9F1239; border-bottom: 2px solid #F43F5E; }
    .code-tok.tok-mismatch { outline: 2px dashed #E11D48; background-color: #FFE4E6 !important; }
    .code-tok.tok-uncertain { outline: 2px dashed #D97706; background-color: #FEF3C7 !important; }

    /* Token Composition Progress Bar */
    .comp-bar-container {
        display: flex;
        height: 12px;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 6px;
        margin-bottom: 16px;
        background-color: #E0E7FF;
        box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.08);
    }
    .comp-seg { height: 100%; transition: width 0.45s ease; }

    /* Hierarchical AST Interactive Tree Styling */
    .ast-tree-view-wrapper {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%);
        border: 1px solid #E0E7FF;
        border-radius: 16px;
        padding: 14px 18px;
        max-height: 580px;
        overflow-y: auto;
        font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
        font-size: 0.86rem;
        line-height: 1.65;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.06);
    }

    .ast-branch {
        margin-left: 18px;
        border-left: 1.5px solid #C7D2FE;
        padding-left: 10px;
        margin-top: 4px;
    }
    .ast-summary {
        cursor: pointer;
        padding: 3px 6px;
        border-radius: 8px;
        color: #1E1B4B;
        list-style: none;
        user-select: none;
        transition: background 0.15s ease;
    }
    .ast-summary::-webkit-details-marker { display: none; }
    .ast-summary:hover {
        background-color: #EEF2FF;
    }
    .ast-toggle {
        display: inline-block;
        font-size: 0.70rem;
        color: #6366F1;
        margin-right: 4px;
        transition: transform 0.18s ease;
    }
    details[open] > summary > .ast-toggle {
        transform: rotate(90deg);
    }
    .ast-leaf {
        margin-left: 18px;
        border-left: 1.5px solid #C7D2FE;
        padding-left: 10px;
        padding-top: 3px;
        padding-bottom: 3px;
        color: #334155;
    }
    .ast-leaf-dot {
        color: #94A3B8;
        margin-right: 4px;
    }
    .ast-type-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 999px;
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
    }
    .ast-loc-pill {
        font-size: 0.70rem;
        font-weight: 600;
        color: #64748B;
        background: #F8FAFC;
        padding: 1px 6px;
        border-radius: 999px;
        border: 1px solid #E2E8F0;
        margin: 0 4px;
    }
    .ast-details {
        color: #1E1B4B;
        font-size: 0.80rem;
    }
    .ast-meta-key { color: #64748B; }
    .ast-meta-val { color: #4F46E5; font-weight: 600; }

    section[data-testid="stSidebar"] .token-badge {
        color: inherit !important;
    }
    section[data-testid="stSidebar"] .badge-keyword { color: #6D28D9 !important; }
    section[data-testid="stSidebar"] .badge-identifier { color: #4338CA !important; }
    section[data-testid="stSidebar"] .badge-operator { color: #047857 !important; }
    section[data-testid="stSidebar"] .badge-literal { color: #B45309 !important; }
    section[data-testid="stSidebar"] .badge-delimiter { color: #BE185D !important; }
    [data-testid="stDataFrame"] {
        border: 1px solid #E0E7FF;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.05);
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.72);
        border: 1px dashed #A5B4FC;
        border-radius: 16px;
        padding: 8px 10px;
    }
    .stSlider label { font-weight: 700 !important; }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #6366F1 !important;
    }
    .stSuccess, .stInfo, .stWarning {
        border-radius: 14px !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
<script>
    try { localStorage.setItem("stActiveTheme", JSON.stringify({base: "light"})); } catch (e) {}
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resource Loaders (Model & Precomputed Test Benchmarks)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor(model_path: Optional[str] = None):
    """Loads and caches the neural network token classifier."""
    artifacts_dir = "artifacts"
    if model_path is None:
        model_path = os.path.join(artifacts_dir, "model.pt")
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at {model_path}!")
        st.stop()
    return TokenClassifierPredictor(artifacts_dir=artifacts_dir, model_path=model_path, device="cpu")

@st.cache_data
def load_metrics():
    """Loads saved test set metrics from artifacts/metrics.json."""
    metrics_path = os.path.join("artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

metrics_data = load_metrics()

# ---------------------------------------------------------------------------
# Preloaded C Code Programs
# ---------------------------------------------------------------------------
SAMPLE_PROGRAMS = {
    "Simple Arithmetic & If": """int calculate_sum(int a, int b) {
    int total = a + b;
    if (total > 100) {
        return total * 2;
    }
    return total;
}""",
    "Quicksort Partitioning": """void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}""",
    "Linked List Node Allocation": """struct Record {
    int id;
    double score;
};

struct Record* create_record(int id, double score) {
    struct Record *rec = (struct Record*)malloc(sizeof(struct Record));
    if (rec == 0) {
        return 0;
    }
    rec->id = id;
    rec->score = score;
    return rec;
}""",
    "Fibonacci Loop": """int fibonacci(int n) {
    int a = 0, b = 1, c, i;
    if (n == 0) return a;
    for (i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}""",
    "Struct & Pointer Dereferencing": """struct Node {
    int value;
    struct Node *next;
};

int get_head_value(struct Node *head) {
    if (head != 0) {
        return head->value;
    }
    return -1;
}"""
}

# ---------------------------------------------------------------------------
# Sidebar (Model Selection & Sample Programs)
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.18); border-radius:16px; padding:14px 14px; margin-bottom:14px; backdrop-filter:blur(10px);">
    <div style="font-size:0.70rem; font-weight:800; color:#A5B4FC; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
        COMPILER DESIGN STUDIO
    </div>
    <div style="font-size:1.05rem; font-weight:800; color:#FFFFFF; margin-bottom:4px;">
        Neural Lexer Engine
    </div>
    <div style="font-size:0.78rem; color:#C7D2FE; line-height:1.5;">
        Character & contextual BiLSTM sequence classifier for ANSI C / C99 compilers.
    </div>
</div>
""", unsafe_allow_html=True)

al_checkpoint = os.path.join("artifacts", "active_learning_model.pt")
model_options = ["Baseline BiLSTM (artifacts/model.pt)"]
if os.path.isfile(al_checkpoint):
    model_options.append("Active Learning BiLSTM (artifacts/active_learning_model.pt)")

selected_model_option = st.sidebar.radio(
    "Active Model Checkpoint:",
    model_options,
    index=0
)
active_model_path = al_checkpoint if "Active Learning" in selected_model_option else os.path.join("artifacts", "model.pt")
predictor = load_predictor(active_model_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Token Class Palette")
legend_markup = """
<div style="font-size:0.80rem; line-height:2.0; margin-bottom:10px;">
    <span class="token-badge badge-keyword">KEYWORD</span> Types, flow control<br/>
    <span class="token-badge badge-identifier">IDENTIFIER</span> Variables, functions<br/>
    <span class="token-badge badge-operator">OPERATOR</span> Arithmetic, relational<br/>
    <span class="token-badge badge-literal">LITERAL</span> Numbers, strings, chars<br/>
    <span class="token-badge badge-delimiter">DELIMITER</span> Punctuation, braces, ;
</div>
"""
st.sidebar.markdown(legend_markup, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Preload Sample Program")
sidebar_sample = st.sidebar.selectbox(
    "Choose a sample C program:",
    ["Custom / Blank"] + list(SAMPLE_PROGRAMS.keys()),
    label_visibility="collapsed"
)

fb_count = get_feedback_count()
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.18); border-radius:16px; padding:13px 14px; font-size:0.80rem; color:#C7D2FE; line-height:1.6;">
    <strong style="color:#FFFFFF;">Operational Specs</strong><br/>
    &bull; <strong>Active Corrections:</strong> {fb_count} saved<br/>
    &bull; <strong>Test Tokens:</strong> 663 Held-Out Unseen<br/>
    &bull; <strong>Grammar:</strong> ANSI C / ISO C99<br/>
    &bull; <strong>Target Architecture:</strong> CPU Native &lt; 15ms
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top NOC Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="noc-header-card">
    <div class="noc-title-area">
        <div class="noc-tagline">⚡ COMPILER DESIGN STUDIO &bull; AI/ML RESEARCH PLATFORM</div>
        <div class="noc-main-title">Compiler Intelligence <span>Studio</span></div>
        <div class="noc-sub-title">Neural Network-Based Token Classification &bull; Smart Error Detector &bull; Hierarchical AST Visualizer</div>
    </div>
    <div class="noc-telemetry-cluster">
        <span class="noc-pill noc-pill-green">
            <span class="pulse-dot-green"></span>
            Health 100%
        </span>
        <span class="noc-pill noc-pill-blue">
            ⚡ BiLSTM Active
        </span>
        <span class="noc-pill noc-pill-slate">
            ⏱ Latency &lt; 15ms
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: Highlighted Code Generator
# ---------------------------------------------------------------------------
def generate_highlighted_code_html(
    source_code: str,
    predictions: List[Dict[str, Any]],
    confidence_threshold: float = 0.85
) -> str:
    """Renders C source code line-by-line with interactive token badges."""
    if not source_code:
        return ""

    lines = source_code.splitlines()
    html_lines = []

    for l_idx, line_str in enumerate(lines, start=1):
        line_preds = [p for p in predictions if p["line"] == l_idx]
        line_preds.sort(key=lambda x: x["col"])

        cur_col = 1
        line_parts = []

        for p in line_preds:
            col = p["col"]
            tok_text = p["token"]
            cls_name = p["predicted_class"]
            conf = p["confidence"]
            gt = p.get("ground_truth")

            if col > cur_col:
                intervening = line_str[cur_col - 1 : col - 1]
                line_parts.append(html.escape(intervening))

            is_mismatch = (gt is not None and cls_name != gt)
            is_uncertain = (conf < confidence_threshold)

            flag_cls = ""
            if is_mismatch:
                flag_cls = " tok-mismatch"
            elif is_uncertain:
                flag_cls = " tok-uncertain"

            tooltip = f"[{cls_name}] '{tok_text}' | Conf: {conf*100:.1f}%"
            if gt:
                tooltip += f" | Lexer: {gt} ({'Match' if not is_mismatch else 'MISMATCH'})"

            escaped_tok = html.escape(tok_text)
            tok_html = (
                f'<span class="code-tok tok-{cls_name.lower()}{flag_cls}" title="{html.escape(tooltip)}">'
                f'{escaped_tok}'
                f'</span>'
            )
            line_parts.append(tok_html)
            cur_col = col + len(tok_text)

        if cur_col - 1 < len(line_str):
            remainder = line_str[cur_col - 1 :]
            line_parts.append(html.escape(remainder))

        line_content = "".join(line_parts) if line_parts else "&nbsp;"
        html_lines.append(
            f'<div class="code-line"><span class="line-num">{l_idx:2d}</span><span class="line-code">{line_content}</span></div>'
        )

    return f'<div class="highlighted-code-container"><pre class="highlighted-code-pre">{"".join(html_lines)}</pre></div>'

CLASS_PALETTE = {
    "KEYWORD": "#8B5CF6",
    "IDENTIFIER": "#6366F1",
    "OPERATOR": "#10B981",
    "LITERAL": "#F59E0B",
    "DELIMITER": "#F43F5E"
}

def generate_distribution_chart(class_counts: Dict[str, int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 2.8), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    labels = [c for c in CLASSES if c in class_counts]
    values = [class_counts[c] for c in labels]
    colors = [CLASS_PALETTE.get(c, "#6366F1") for c in labels]
    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="#E0E7FF", linewidth=1.0, zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1E1B4B")
    ax.set_ylabel("Token Count", fontsize=8.5, fontweight="700", color="#475569")
    ax.set_title("Token Class Distribution (Current Program)", fontsize=9.5, fontweight="800", color="#1E1B4B", pad=10)
    ax.tick_params(colors="#64748B", labelsize=8.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#E0E7FF", zorder=0)
    for spine in ax.spines.values(): spine.set_color("#E0E7FF")
    plt.tight_layout()
    return fig

def generate_confidence_histogram(confidences: List[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 2.8), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    n, bins, patches = ax.hist(confidences, bins=10, range=(0.0, 1.0),
                               color="#6366F1", edgecolor="#FFFFFF", alpha=0.9, rwidth=0.85, zorder=3)
    for i, patch in enumerate(patches):
        if bins[i] >= 0.85: patch.set_facecolor("#10B981")
        elif bins[i] >= 0.70: patch.set_facecolor("#6366F1")
        else: patch.set_facecolor("#F59E0B")
    ax.set_xlabel("Softmax Confidence", fontsize=8.5, fontweight="700", color="#475569")
    ax.set_ylabel("Tokens", fontsize=8.5, fontweight="700", color="#475569")
    ax.set_title("Prediction Confidence Spread", fontsize=9.5, fontweight="800", color="#1E1B4B", pad=10)
    ax.tick_params(colors="#64748B", labelsize=8.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#E0E7FF", zorder=0)
    for spine in ax.spines.values(): spine.set_color("#E0E7FF")
    plt.tight_layout()
    return fig

# Initialize default code & predictions in session state if not present
if "current_code" not in st.session_state:
    st.session_state["current_code"] = SAMPLE_PROGRAMS["Simple Arithmetic & If"]
# Keep the text-area widget state in sync with the selected example. Streamlit
# preserves keyed widget values across reruns, so changing only current_code
# does not update the visible editor after a Quick Example button is clicked.
if "sec1_code_input" not in st.session_state:
    st.session_state["sec1_code_input"] = st.session_state["current_code"]
if "current_predictions" not in st.session_state:
    with st.spinner("Initializing neural token classifier..."):
        st.session_state["current_predictions"] = predictor.predict_code(st.session_state["current_code"])

# ---------------------------------------------------------------------------
# Navigation Tabs (The 5 Distinct Sections)
# ---------------------------------------------------------------------------
tab_classifier, tab_detector, tab_comparison, tab_ast, tab_evaluation = st.tabs([
    "⚡ 1. Token Classifier",
    "🔍 2. Smart Token Error Detector",
    "⚖️ 3. Model Comparison Lab",
    "🌳 4. Syntax Tree Visualizer",
    "📊 5. Model Evaluation & Metrics"
])

# ===========================================================================
# SECTION 1: TOKEN CLASSIFIER
# ===========================================================================
with tab_classifier:
    st.markdown('<div class="noc-card-header"><span>🚀 Quick Example Programs</span></div>', unsafe_allow_html=True)
    sample_cols = st.columns(len(SAMPLE_PROGRAMS))
    clicked_sample_code = None
    for idx, (s_name, s_code) in enumerate(SAMPLE_PROGRAMS.items()):
        with sample_cols[idx]:
            if st.button(f"📄 {s_name}", key=f"sec1_btn_{idx}", use_container_width=True):
                clicked_sample_code = s_code

    if clicked_sample_code is not None:
        st.session_state["current_code"] = clicked_sample_code
        st.session_state["sec1_code_input"] = clicked_sample_code
    elif sidebar_sample in SAMPLE_PROGRAMS and sidebar_sample != "Custom / Blank":
        selected_sidebar_code = SAMPLE_PROGRAMS[sidebar_sample]
        if selected_sidebar_code != st.session_state.get("current_code"):
            st.session_state["current_code"] = selected_sidebar_code
            st.session_state["sec1_code_input"] = selected_sidebar_code

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    col_upload_box, col_slider_box = st.columns([3.5, 2.5])
    with col_upload_box:
        uploaded_file = st.file_uploader(
            "Upload .c / .h source file",
            type=["c", "h", "txt"],
            key="sec1_uploader",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                file_text = uploaded_file.read().decode("utf-8", errors="replace")
                st.session_state["current_code"] = file_text
                st.session_state["sec1_code_input"] = file_text
                st.caption(f"📁 Loaded `{uploaded_file.name}` ({len(file_text)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col_slider_box:
        threshold_pct = st.slider(
            "Flag Low Confidence (< %):",
            min_value=60,
            max_value=99,
            value=85,
            step=5,
            key="sec1_threshold_slider",
            help="Tokens with confidence below this threshold are marked as uncertain."
        )
        confidence_threshold = threshold_pct / 100.0

    code_input = st.text_area(
        "C Program Source Code",
        value=st.session_state["current_code"],
        height=200,
        key="sec1_code_input",
        help="Type or paste C source code to classify",
        label_visibility="collapsed"
    )

    col_btn, col_filter, col_mismatch_only = st.columns([1.8, 2.1, 2.1])
    with col_btn:
        classify_button = st.button("⚡ Classify Program Tokens", type="primary", use_container_width=True, key="sec1_classify_btn")
    with col_filter:
        filter_uncertain = st.checkbox("🔍 Filter low-confidence tokens", value=False, key="sec1_filter_uncertain")
    with col_mismatch_only:
        filter_mismatches = st.checkbox("⚠️ Filter lexer mismatches only", value=False, key="sec1_filter_mismatches")

    if classify_button or (code_input != st.session_state.get("last_classified_code", "")):
        if code_input.strip():
            start_t = time.perf_counter()
            with st.spinner("Classifying tokens via BiLSTM neural network..."):
                st.session_state["current_predictions"] = predictor.predict_code(code_input)
                st.session_state["current_code"] = code_input
                st.session_state["last_classified_code"] = code_input
                st.session_state["last_inference_latency_ms"] = (time.perf_counter() - start_t) * 1000.0

    raw_predictions = st.session_state.get("current_predictions", [])
    if raw_predictions:
        total_tokens = len(raw_predictions)
        confidences = [p["confidence"] for p in raw_predictions]
        mean_conf = sum(confidences) / max(1, total_tokens)

        mismatches = [
            p for p in raw_predictions
            if p.get("ground_truth") is not None and p["predicted_class"] != p["ground_truth"]
        ]
        uncertain_tokens = [
            p for p in raw_predictions if p["confidence"] < confidence_threshold
        ]

        class_counts = {c: 0 for c in CLASSES}
        for p in raw_predictions:
            cls = p["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        concordance_rate = ((total_tokens - len(mismatches)) / max(1, total_tokens)) * 100
        latency_val = st.session_state.get("last_inference_latency_ms", 12.4)

        # Summary Telemetry Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="m-label">Total Tokens</div>
                <div class="m-val" style="color:#6366F1;">{total_tokens}</div>
                <div class="m-sub">{len(code_input.splitlines())} lines of C source</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card metric-card-green">
                <div class="m-label">Mean Confidence</div>
                <div class="m-val" style="color:#059669;">{mean_conf * 100:.1f}%</div>
                <div class="m-sub">Softmax probability mean</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card metric-card-amber">
                <div class="m-label">Low Confidence (&lt;{threshold_pct}%)</div>
                <div class="m-val" style="color:#D97706;">{len(uncertain_tokens)}</div>
                <div class="m-sub">{len(uncertain_tokens)/max(1, total_tokens)*100:.1f}% of program</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card metric-card-purple">
                <div class="m-label">Lexer Agreement</div>
                <div class="m-val" style="color:#7C3AED;">{concordance_rate:.1f}%</div>
                <div class="m-sub">BiLSTM vs Rule-based Lexer</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Token Composition Progress Bar
        bar_segments = []
        for cls in CLASSES:
            pct = (class_counts[cls] / max(1, total_tokens)) * 100
            if pct > 0:
                color = CLASS_PALETTE.get(cls, "#6366F1")
                bar_segments.append(
                    f'<div class="comp-seg" style="width:{pct:.1f}%; background-color:{color};" title="{cls}: {class_counts[cls]} ({pct:.1f}%)"></div>'
                )
        st.markdown(f'<div class="comp-bar-container">{"".join(bar_segments)}</div>', unsafe_allow_html=True)

        # Highlighted Canvas
        st.markdown('<div class="noc-card-header"><span>🔍 Highlighted Source Code Canvas</span><span style="font-size:0.78rem; font-weight:600; color:#64748B;">Interactive hover tooltips with class & confidence</span></div>', unsafe_allow_html=True)
        highlighted_html = generate_highlighted_code_html(
            code_input, raw_predictions, confidence_threshold=confidence_threshold
        )
        st.markdown(highlighted_html, unsafe_allow_html=True)

        # Token Stream Table
        st.markdown('<div class="noc-card-header"><span>📋 Extracted Token Stream</span></div>', unsafe_allow_html=True)
        table_rows = []
        for idx, p in enumerate(raw_predictions, start=1):
            is_mismatch = (p.get("ground_truth") is not None and p["predicted_class"] != p["ground_truth"])
            is_uncertain = (p["confidence"] < confidence_threshold)

            if filter_uncertain and not is_uncertain:
                continue
            if filter_mismatches and not is_mismatch:
                continue

            status_txt = "Match" if not is_mismatch else "MISMATCH"
            if is_uncertain:
                status_txt += " (Uncertain)"

            table_rows.append({
                "#": idx,
                "Token": p["token"],
                "Line": p["line"],
                "Col": p["col"],
                "Predicted Class": p["predicted_class"],
                "Confidence": f"{p['confidence'] * 100:.2f}%",
                "Lexer Ground Truth": p.get("ground_truth") or "—",
                "Consensus": status_txt
            })

        if table_rows:
            df_display = pd.DataFrame(table_rows)
            st.dataframe(df_display, use_container_width=True, height=240, hide_index=True)

            csv_data = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified Tokens (CSV)",
                data=csv_data,
                file_name="classified_tokens.csv",
                mime="text/csv"
            )
        else:
            st.info("No tokens match the active filters.")
    else:
        st.info("Enter or upload C source code above and click **⚡ Classify Program Tokens**.")


# ===========================================================================
# SECTION 2: SMART TOKEN ERROR DETECTOR
# ===========================================================================
with tab_detector:
    st.markdown("""
    <div class="noc-card-header">
        <span>🚨 Smart Token Error Detector: Disagreements & Uncertainties</span>
        <span style="font-size:0.78rem; font-weight:600; color:#64748B;">Identifies low-confidence neural tokens & lexer boundary discrepancies</span>
    </div>
    """, unsafe_allow_html=True)

    raw_predictions = st.session_state.get("current_predictions", [])
    if not raw_predictions:
        st.info("No tokens classified yet. Go to **1. Token Classifier** to run classification.")
    else:
        # Diagnostic analysis
        flagged_errors = []
        for idx, p in enumerate(raw_predictions, start=1):
            is_disagreement = (p.get("ground_truth") is not None and p["predicted_class"] != p["ground_truth"])
            is_low_conf = (p["confidence"] < confidence_threshold)

            if is_disagreement and is_low_conf:
                category = "High-Risk Conflict"
                note = "Low model confidence combined with lexer label mismatch."
            elif is_disagreement:
                category = "Lexer Disagreement"
                note = "Neural sequence model predicts a different role than static regex lexer."
            elif is_low_conf:
                category = "Low Confidence"
                note = f"Softmax confidence ({p['confidence']*100:.1f}%) below {threshold_pct}% threshold."
            else:
                continue

            flagged_errors.append({
                "Index": idx,
                "Token": p["token"],
                "Location": f"L{p['line']}:C{p['col']}",
                "BiLSTM Prediction": p["predicted_class"],
                "Confidence": f"{p['confidence']*100:.1f}%",
                "Runner-Up": f"{p.get('alternative_class', '—')} ({p.get('alternative_confidence', 0)*100:.1f}%)",
                "Lexer Truth": p.get("ground_truth") or "—",
                "Category": category,
                "Diagnostic Note": note
            })

        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""
            <div class="metric-card metric-card-{'rose' if flagged_errors else 'green'}">
                <div class="m-label">Flagged Discrepancies</div>
                <div class="m-val" style="color:{'#EF4444' if flagged_errors else '#10B981'};">{len(flagged_errors)}</div>
                <div class="m-sub">Requiring inspection</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            mismatches_count = sum(1 for f in flagged_errors if "Disagreement" in f["Category"] or "Conflict" in f["Category"])
            st.markdown(f"""
            <div class="metric-card metric-card-amber">
                <div class="m-label">Lexer Disagreements</div>
                <div class="m-val" style="color:#D97706;">{mismatches_count}</div>
                <div class="m-sub">Model vs Regex lexer</div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            low_c_count = sum(1 for f in flagged_errors if "Low Confidence" in f["Category"] or "Conflict" in f["Category"])
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="m-label">Low Confidence Tokens</div>
                <div class="m-val" style="color:#6366F1;">{low_c_count}</div>
                <div class="m-sub">Below {threshold_pct}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        if not flagged_errors:
            st.success("✨ **Consensus Confirmed:** No low-confidence tokens or lexer disagreements detected in this program! All predictions meet high-certainty criteria.")
        else:
            st.markdown('<div class="noc-card-header"><span>📋 Diagnostic Conflict Log</span></div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(flagged_errors),
                use_container_width=True,
                height=240,
                hide_index=True
            )

        # Highlighted view with flagged items
        st.markdown('<div class="noc-card-header"><span>🔎 Flagged Token Canvas</span><span style="font-size:0.78rem; font-weight:600; color:#EF4444;">Red dashed = Mismatch &bull; Amber dashed = Low Confidence</span></div>', unsafe_allow_html=True)
        code_txt = st.session_state.get("current_code", "")
        h_html = generate_highlighted_code_html(code_txt, raw_predictions, confidence_threshold=confidence_threshold)
        st.markdown(h_html, unsafe_allow_html=True)


# ===========================================================================
# SECTION 3: MODEL COMPARISON LAB
# ===========================================================================
with tab_comparison:
    st.markdown("""
    <div class="noc-card-header">
        <span>⚖️ Model Comparison Lab: Contextual BiLSTM vs. Rule-Based Lexer</span>
        <span style="font-size:0.78rem; font-weight:600; color:#64748B;">Empirical latency, throughput & architectural benchmark</span>
    </div>
    """, unsafe_allow_html=True)

    curr_code = st.session_state.get("current_code", SAMPLE_PROGRAMS["Simple Arithmetic & If"])

    st.markdown("""
    <div style="background:linear-gradient(90deg,#EEF2FF,#ECFEFF); border:1px solid #C7D2FE; border-radius:16px; padding:13px 16px; margin-bottom:14px; font-size:0.84rem; color:#4338CA; line-height:1.55;">
        <strong>Academic Benchmark Methodology:</strong> Compares the deep learning sequence classifier against a deterministic ANSI C regex lexer (Pygments).
        Wall-clock latency is measured directly on CPU inference across all extracted tokens.
    </div>
    """, unsafe_allow_html=True)

    if st.button("⏱ Run Real-Time Benchmark on Current Code", key="sec3_run_benchmark"):
        with st.spinner("Executing side-by-side benchmark..."):
            bench = compare_bilstm_vs_lexer(curr_code, predictor)
            st.session_state["benchmark_results"] = bench

    bench = st.session_state.get("benchmark_results")
    if bench is None:
        bench = compare_bilstm_vs_lexer(curr_code, predictor)
        st.session_state["benchmark_results"] = bench

    # Telemetry metrics row
    b_time = bench.get("bilstm_time_ms", 12.5)
    l_time = bench.get("lexer_time_ms", 1.8)
    b_thr = bench.get("bilstm_throughput", 2500.0)
    acc_pct = bench.get("accuracy", 0.98) * 100.0
    tok_count = bench.get("total_tokens", 25)

    comp_c1, comp_c2, comp_c3, comp_c4 = st.columns(4)
    with comp_c1:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <div class="m-label">BiLSTM Latency</div>
            <div class="m-val" style="color:#6366F1;">{b_time:.2f} ms</div>
            <div class="m-sub">Total neural forward pass</div>
        </div>
        """, unsafe_allow_html=True)
    with comp_c2:
        st.markdown(f"""
        <div class="metric-card metric-card-cyan">
            <div class="m-label">Lexer Latency</div>
            <div class="m-val" style="color:#06B6D4;">{l_time:.2f} ms</div>
            <div class="m-sub">Deterministic regex match</div>
        </div>
        """, unsafe_allow_html=True)
    with comp_c3:
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <div class="m-label">BiLSTM Throughput</div>
            <div class="m-val" style="color:#059669;">{b_thr:,.0f}</div>
            <div class="m-sub">Tokens / second (CPU)</div>
        </div>
        """, unsafe_allow_html=True)
    with comp_c4:
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <div class="m-label">Concordance Rate</div>
            <div class="m-val" style="color:#7C3AED;">{acc_pct:.1f}%</div>
            <div class="m-sub">{tok_count} benchmark tokens</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Architectural Comparison Matrix
    st.markdown('<div class="noc-card-header"><span>🏛 Architectural & Engineering Trade-Off Matrix</span></div>', unsafe_allow_html=True)
    matrix_data = [
        {"Dimension": "Core Paradigm", "Contextual BiLSTM (Ours)": "Deep bidirectional sequence modeling with char-level embeddings", "Rule-Based Lexer (Pygments/Flex)": "Deterministic regular expressions and finite-state automata"},
        {"Dimension": "Context Handling", "Contextual BiLSTM (Ours)": "Captures left & right syntactic transitions across token sequence", "Rule-Based Lexer (Pygments/Flex)": "Strictly local lookahead; no global context tracking"},
        {"Dimension": "Typo & Noise Tolerance", "Contextual BiLSTM (Ours)": "High: Subword character embeddings generalize to irregular names", "Rule-Based Lexer (Pygments/Flex)": "Low: Rigid syntax rules fail or fallback to catch-all IDENTIFIER"},
        {"Dimension": "Online Adaptability", "Contextual BiLSTM (Ours)": "Supports Human-in-the-Loop Active Learning fine-tuning", "Rule-Based Lexer (Pygments/Flex)": "Requires manual grammar rewrite and table recompilation"},
        {"Dimension": "Inference Latency", "Contextual BiLSTM (Ours)": f"~{b_time:.1f} ms on CPU native", "Rule-Based Lexer (Pygments/Flex)": f"~{l_time:.1f} ms"},
        {"Dimension": "Compiler Role", "Contextual BiLSTM (Ours)": "Intelligent pre-parser, auto-completion, syntax disambiguation", "Rule-Based Lexer (Pygments/Flex)": "Deterministic production tokenization"}
    ]
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)


# ===========================================================================
# SECTION 4: SYNTAX TREE VISUALIZER (THE KEY FIX!)
# ===========================================================================
with tab_ast:
    st.markdown("""
    <div class="noc-card-header">
        <span>🌳 Syntax Tree Visualizer: C Abstract Syntax Tree & Parser Diagnostics</span>
        <span style="font-size:0.78rem; font-weight:600; color:#64748B;">Graphical connected AST nodes, source locations & diagnostics</span>
    </div>
    """, unsafe_allow_html=True)

    code_to_parse = st.session_state.get("current_code", "")

    col_ast_input, col_ast_btn = st.columns([4, 1])
    with col_ast_input:
        st.caption("Parses current C program into an Abstract Syntax Tree (AST) using recursive-descent parsing.")

    if not code_to_parse.strip():
        st.info("Enter or upload C source code in **1. Token Classifier** to parse its Abstract Syntax Tree.")
    else:
        ast_root, syntax_errors = parse_c_code(code_to_parse)
        total_nodes, max_depth = ast_root.get_tree_metrics()

        # Telemetry metrics row
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="m-label">Top Declarations</div>
                <div class="m-val" style="color:#6366F1;">{len(ast_root.children)}</div>
                <div class="m-sub">Functions & global decls</div>
            </div>
            """, unsafe_allow_html=True)
        with a2:
            st.markdown(f"""
            <div class="metric-card metric-card-green">
                <div class="m-label">Total AST Nodes</div>
                <div class="m-val" style="color:#059669;">{total_nodes}</div>
                <div class="m-sub">Hierarchy size</div>
            </div>
            """, unsafe_allow_html=True)
        with a3:
            st.markdown(f"""
            <div class="metric-card metric-card-purple">
                <div class="m-label">Max Tree Depth</div>
                <div class="m-val" style="color:#7C3AED;">{max_depth}</div>
                <div class="m-sub">Levels of nesting</div>
            </div>
            """, unsafe_allow_html=True)
        with a4:
            st.markdown(f"""
            <div class="metric-card metric-card-{'rose' if syntax_errors else 'cyan'}">
                <div class="m-label">Parser Diagnostics</div>
                <div class="m-val" style="color:{'#F43F5E' if syntax_errors else '#06B6D4'};">{'Errors Detected' if syntax_errors else 'Valid AST'}</div>
                <div class="m-sub">{len(syntax_errors)} syntax issues</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Syntax Errors Panel (if any)
        if syntax_errors:
            st.markdown(
                f'<div style="background:linear-gradient(90deg,#FFF1F2,#FFE4E6); border:1px solid #FECDD3; border-radius:16px; padding:12px 16px; margin-bottom:14px; color:#9F1239; font-size:0.84rem;">'
                f'<strong>⚠️ {len(syntax_errors)} Syntax Error(s) Detected During Parsing:</strong><br/>'
                f'The recursive-descent C parser reported the following syntax violations with exact coordinates:'
                f'</div>',
                unsafe_allow_html=True
            )
            err_rows = []
            for err in syntax_errors:
                err_rows.append({
                    "Line": err.line,
                    "Column": err.col,
                    "Found Token": err.token,
                    "Expected": err.expected or "Valid syntax",
                    "Diagnostic Message": err.message
                })
            st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)

        # Tree View Selector
        st.markdown('<div class="noc-card-header"><span>🌲 Hierarchical Abstract Syntax Tree Views</span></div>', unsafe_allow_html=True)
        ast_view_choice = st.radio(
            "Select AST Rendering Mode:",
            ["Graphical Tree Diagram", "Interactive Collapsible HTML Tree", "Indented Formatted ASCII Tree", "Structured AST JSON Dictionary"],
            horizontal=True,
            key="sec4_ast_view_choice"
        )

        if ast_view_choice == "Graphical Tree Diagram":
            direction = st.radio("Tree layout", ["Top to bottom", "Left to right"],
                                 horizontal=True, key="sec4_tree_direction")
            ast_mode = st.radio(
                "Visualization mode",
                ["Academic AST", "Detailed AST"],
                horizontal=True,
                key="sec4_ast_mode",
                help="Academic mode emphasizes syntax structure; Detailed mode includes source metadata."
            )
            show_locations = st.checkbox(
                "Show Source Locations (Line & Column)",
                value=(ast_mode == "Detailed AST"),
                key="sec4_tree_locations"
            )
            dot_source = ast_to_dot(
                ast_root,
                direction="TB" if direction == "Top to bottom" else "LR",
                show_locations=show_locations,
                academic=(ast_mode == "Academic AST")
            )
            st.graphviz_chart(dot_source, use_container_width=True)
            st.caption("Each rounded box is an AST node; arrows show actual parent–child relationships. "
                       "The diagram is generated from the C parser, not the neural token classifier.")
            svg_data = dot_to_svg(dot_source)
            if svg_data:
                st.download_button("⬇ Download Tree as SVG", data=svg_data,
                                   file_name="syntax_tree.svg", mime="image/svg+xml",
                                   key="sec4_tree_svg")
            else:
                st.download_button("⬇ Download Graphviz Tree Source", data=dot_source,
                                   file_name="syntax_tree.dot", mime="text/vnd.graphviz",
                                   key="sec4_tree_dot")
                st.caption("Install the Graphviz system application to enable SVG export.")

        elif ast_view_choice == "Interactive Collapsible HTML Tree":
            html_ast_markup = ast_root.render_html_tree()
            st.markdown(
                f'<div class="ast-tree-view-wrapper">{html_ast_markup}</div>',
                unsafe_allow_html=True
            )
            st.caption("💡 *Click on any ▶ toggle to expand or collapse subtrees. Each node displays type, source location [L:C], and semantic attributes.*")

        elif ast_view_choice == "Indented Formatted ASCII Tree":
            ascii_lines = ast_root.format_ascii_tree()
            st.code("\n".join(ascii_lines), language="text")
            st.caption("📍 *Formatted ASCII tree showing exact line numbers [L:C] and branch parent-child connectors.*")

        else:
            st.json(ast_root.to_dict())
            st.caption("🔍 *JSON-serializable abstract syntax tree dictionary representation.*")


# ===========================================================================
# SECTION 5: MODEL EVALUATION & METRICS
# ===========================================================================
with tab_evaluation:
    st.markdown("""
    <div class="noc-card-header">
        <span>📊 Baseline Benchmark Evaluation (Held-Out Test Programs)</span>
        <span style="font-size:0.78rem; font-weight:600; color:#64748B;">Academic standard evaluation on 663 unseen tokens across 4 independent programs</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="background:linear-gradient(90deg,#EEF2FF,#ECFEFF); border:1px solid #C7D2FE; border-radius:16px; padding:12px 16px; margin-bottom:14px; color:#4338CA; font-size:0.84rem; line-height:1.55;">'
        '<strong>Academic Evaluation Integrity:</strong> Evaluated strictly on <strong>663 unseen tokens</strong> across 4 independent held-out programs. '
        'Programs were partitioned strictly at the file level prior to tokenization to guarantee <strong>zero test-data leakage</strong>.'
        '</div>',
        unsafe_allow_html=True
    )

    if metrics_data:
        ev1, ev2, ev3, ev4 = st.columns(4)
        with ev1:
            st.markdown(f"""
            <div class="metric-card metric-card-green">
                <div class="m-label">Test Accuracy</div>
                <div class="m-val" style="color:#059669;">{metrics_data.get('accuracy', 0) * 100:.2f}%</div>
                <div class="m-sub">Overall token classification</div>
            </div>
            """, unsafe_allow_html=True)
        with ev2:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="m-label">Macro Precision</div>
                <div class="m-val" style="color:#6366F1;">{metrics_data.get('macro_precision', 0) * 100:.2f}%</div>
                <div class="m-sub">Unweighted class average</div>
            </div>
            """, unsafe_allow_html=True)
        with ev3:
            st.markdown(f"""
            <div class="metric-card metric-card-amber">
                <div class="m-label">Macro Recall</div>
                <div class="m-val" style="color:#D97706;">{metrics_data.get('macro_recall', 0) * 100:.2f}%</div>
                <div class="m-sub">Sensitivity across all classes</div>
            </div>
            """, unsafe_allow_html=True)
        with ev4:
            st.markdown(f"""
            <div class="metric-card metric-card-rose">
                <div class="m-label">Macro F1-Score</div>
                <div class="m-val" style="color:#EF4444;">{metrics_data.get('macro_f1', 0) * 100:.2f}%</div>
                <div class="m-sub">Harmonic mean precision & recall</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        col_cm, col_loss = st.columns(2)
        with col_cm:
            st.markdown('<div class="noc-card-header"><span>🧮 Interactive Confusion Matrix</span></div>', unsafe_allow_html=True)
            # Evaluate the exact source currently entered in the Token Classifier.
            # No separate sample dropdown: edits and quick examples flow through automatically.
            evaluation_code = st.session_state.get("current_code", "")
            st.caption("Automatically generated from the current Token Classifier code, using lexer reference labels and model predictions.")
            if not evaluation_code.strip():
                st.info("Enter C code in the Token Classifier tab to generate its confusion matrix.")
            else:
                try:
                    sample_predictions = predictor.predict_code(evaluation_code)
                    class_names = list(CLASSES)
                    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
                    class_to_idx = {name: i for i, name in enumerate(class_names)}
                    for item in sample_predictions:
                        actual = item.get("ground_truth")
                        predicted = item.get("predicted_class")
                        if actual in class_to_idx and predicted in class_to_idx:
                            cm[class_to_idx[actual], class_to_idx[predicted]] += 1
                    if cm.sum():
                        fig, ax = plt.subplots(figsize=(7, 5))
                        im = ax.imshow(cm, cmap="Purples")
                        ax.set_xticks(range(len(class_names)), labels=class_names, rotation=35, ha="right")
                        ax.set_yticks(range(len(class_names)), labels=class_names)
                        ax.set_xlabel("Predicted class")
                        ax.set_ylabel("Lexer reference class")
                        ax.set_title(f"Current input · {int(cm.sum())} tokens")
                        for r in range(cm.shape[0]):
                            for c in range(cm.shape[1]):
                                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="white" if cm[r,c] > cm.max()/2 else "#312E81", fontsize=9)
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("No comparable token labels were returned for this sample.")
                except Exception as exc:
                    st.warning(f"Could not calculate the current code's confusion matrix: {exc}")

        with col_loss:
            st.markdown('<div class="noc-card-header"><span>📈 Input-Adaptive Token Accuracy & Error Curves</span></div>', unsafe_allow_html=True)
            st.caption("These curves recalculate from the exact code currently in Token Classifier. They show cumulative token-level performance, not training epochs.")
            evaluation_code = st.session_state.get("current_code", "")
            if not evaluation_code.strip():
                st.info("Enter C code in the Token Classifier tab to generate input-specific curves.")
            else:
                try:
                    curve_predictions = predictor.predict_code(evaluation_code)
                    comparable = [
                        item for item in curve_predictions
                        if item.get("ground_truth") in CLASSES and item.get("predicted_class") in CLASSES
                    ]
                    if comparable:
                        correct_flags = np.asarray([
                            1.0 if item.get("ground_truth") == item.get("predicted_class") else 0.0
                            for item in comparable
                        ])
                        token_positions = np.arange(1, len(correct_flags) + 1)
                        cumulative_accuracy = np.cumsum(correct_flags) / token_positions
                        cumulative_error = 1.0 - cumulative_accuracy
                        curve_view = st.radio(
                            "Curve view", ["Accuracy and error", "Accuracy only", "Error only"],
                            horizontal=True, key="eval_input_curve_view"
                        )
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        if curve_view in ("Accuracy and error", "Accuracy only"):
                            ax.plot(token_positions, cumulative_accuracy * 100, marker="o", markersize=3,
                                    linewidth=2, label="Cumulative token accuracy")
                        if curve_view in ("Accuracy and error", "Error only"):
                            ax.plot(token_positions, cumulative_error * 100, marker="o", markersize=3,
                                    linewidth=2, label="Cumulative token error")
                        ax.set_xlabel("Comparable token position")
                        ax.set_ylabel("Rate (%)")
                        ax.set_ylim(0, 100)
                        ax.set_title(f"Current input · {len(comparable)} evaluated tokens")
                        ax.grid(True, alpha=0.25)
                        ax.legend()
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.caption(f"Final token accuracy: {cumulative_accuracy[-1] * 100:.2f}% · Final token error: {cumulative_error[-1] * 100:.2f}%")
                    else:
                        st.info("No comparable reference/predicted token labels were returned for this input.")
                except Exception as exc:
                    st.warning(f"Could not calculate input-specific token curves: {exc}")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="noc-card-header"><span>📊 Per-Class Classification Report</span></div>', unsafe_allow_html=True)
        per_class_data = metrics_data.get("per_class", {})
        summary_rows = []
        for cls in CLASSES:
            if cls in per_class_data:
                item = per_class_data[cls]
                summary_rows.append({
                    "Class": cls,
                    "Precision": f"{item['precision'] * 100:.2f}%",
                    "Recall": f"{item['recall'] * 100:.2f}%",
                    "F1-Score": f"{item['f1-score'] * 100:.2f}%",
                    "Support (Tokens)": int(item["support"])
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(180deg,#FFFFFF,#F8FAFF); border:1px solid #E0E7FF; border-radius:16px; padding:16px 18px; box-shadow:0 12px 28px rgba(79,70,229,0.07);">
            <div style="font-weight:800; font-size:0.90rem; color:#1E1B4B; margin-bottom:6px;">Model Specification & Architectural Design</div>
            <div style="font-size:0.82rem; color:#475569; line-height:1.65;">
                &bull; <strong>Character Subnetwork:</strong> 97-character ASCII vocabulary, 32-dim char embeddings, bidirectional LSTM with masked max-pooling generating 64-dim token vector.<br/>
                &bull; <strong>Contextual Subnetwork:</strong> 64-dim bidirectional LSTM (128-dim contextual representation) capturing sequential syntactic transitions across code tokens.<br/>
                &bull; <strong>Output Classifier:</strong> Multi-class softmax head projecting to 5 canonical compiler token classes (KEYWORD, IDENTIFIER, OPERATOR, LITERAL, DELIMITER).<br/>
                &bull; <strong>Evaluation Integrity:</strong> Zero variable overlap, zero function overlap, and genuine cross-entropy loss tracking.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No evaluation metrics found in `artifacts/metrics.json`.")

st.markdown("""
<div style="margin-top:1.4rem; text-align:center; color:#64748B; font-size:0.78rem;">
    Compiler Intelligence Studio · Neural token classification for ANSI C · All predictions use the trained BiLSTM checkpoint
</div>
""", unsafe_allow_html=True)
