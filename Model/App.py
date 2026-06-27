import streamlit as st
import torch
import tiktoken
import re
import base64
from pathlib import Path

from Previous_chapters import GPTModel, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2

st.set_page_config(page_title="LLM Smart Mirror", layout="centered")

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("background_image.jpg")

st.markdown(f"""
    <style>
    /* ── Page & Background ── */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        animation: fadeIn 1s ease-in-out;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* ── Reduce default Streamlit top padding ── */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 780px;
    }}

    /* ── Headings ── */
    h1 {{
        font-size: 1.6rem !important;
        margin-bottom: 0.1rem !important;
        color: #1e1e1e !important;
        text-shadow: 0 0 4px rgba(255,255,255,0.7);
    }}
    h2, h3,
    .stMarkdown h2, .stMarkdown h3 {{
        font-size: 1rem !important;
        color: #1e1e1e !important;
        text-shadow: 0 0 3px rgba(255,255,255,0.6);
        margin-bottom: 0.2rem !important;
    }}

    /* ── Subtitle / description text ── */
    .subtitle {{
        font-size: 0.82rem;
        color: #2e2e2e;
        margin-bottom: 0.6rem;
        background: rgba(255,255,255,0.45);
        border-radius: 8px;
        padding: 4px 10px;
        display: inline-block;
    }}

    /* ── Text areas ── */
    textarea, .stTextArea textarea {{
        background-color: rgba(0, 0, 0, 0.72) !important;
        color: #f0f0f0 !important;
        border: 1px solid #6a8f6a !important;
        border-radius: 8px !important;
        font-size: 0.84rem !important;
        line-height: 1.45 !important;
    }}

    /* Tighten textarea labels */
    .stTextArea label p {{
        font-size: 0.82rem !important;
        font-weight: 600;
        color: #111 !important;
        text-shadow: 0 0 3px rgba(255,255,255,0.8);
        margin-bottom: 2px !important;
    }}

    /* ── Generate button ── */
    .stButton > button {{
        background-color: #a3b18a;
        color: #1a1a1a;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 0.4rem 1.6rem;
        border: none;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        width: 100%;
    }}
    .stButton > button:hover {{
        background-color: #588157;
        color: #fff;
        box-shadow: 0 4px 14px rgba(0,0,0,0.26);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: scale(0.97);
    }}

    /* ── Expander (Advanced Options) — all Streamlit versions ── */
    /* Header wrapper */
    [data-testid="stExpander"] details {{
        background-color: rgba(25, 25, 25, 0.78) !important;
        border: 1px solid #5a7a5a !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] details summary {{
        background-color: rgba(25, 25, 25, 0.78) !important;
        border-radius: 8px !important;
        padding: 0.4rem 0.9rem !important;
        list-style: none;
    }}
    /* Force the label text to always be light & visible */
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary,
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span {{
        color: #d4e8c2 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
    }}
    /* Arrow / chevron icon */
    [data-testid="stExpander"] details summary svg {{
        fill: #d4e8c2 !important;
        stroke: #d4e8c2 !important;
    }}
    /* Hover state */
    [data-testid="stExpander"] details summary:hover,
    .streamlit-expanderHeader:hover {{
        background-color: rgba(55, 75, 55, 0.85) !important;
    }}
    [data-testid="stExpander"] details summary:hover p,
    [data-testid="stExpander"] details summary:hover span,
    .streamlit-expanderHeader:hover p,
    .streamlit-expanderHeader:hover span {{
        color: #ffffff !important;
    }}
    /* Expander content area */
    [data-testid="stExpander"] details[open] > div,
    .streamlit-expanderContent {{
        background-color: rgba(20, 20, 20, 0.72) !important;
        border: 1px solid #4a6a4a !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 0.6rem 0.9rem 0.8rem !important;
    }}
    /* Slider label text inside expander */
    [data-testid="stExpander"] label p,
    [data-testid="stExpander"] .stSlider label p,
    .streamlit-expanderContent label p {{
        color: #c8ddb0 !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }}
    /* Slider value readout */
    [data-testid="stExpander"] [data-testid="stSlider"] span,
    [data-testid="stExpander"] .stSlider span {{
        color: #b0cc90 !important;
    }}
    /* Slider track & thumb */
    [data-testid="stExpander"] input[type=range] {{
        accent-color: #588157 !important;
    }}

    /* ── Response box ── */
    .response-box {{
        background-color: rgba(15, 15, 15, 0.78);
        border: 1px solid #5a8a5a;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        color: #e8f0e0;
        font-size: 0.84rem;
        line-height: 1.55;
        min-height: 70px;
        white-space: pre-wrap;
        word-break: break-word;
    }}
    .response-label {{
        font-size: 0.82rem;
        font-weight: 700;
        color: #1e1e1e;
        text-shadow: 0 0 4px rgba(255,255,255,0.8);
        margin-bottom: 3px;
    }}

    /* ── Spinner text ── */
    .stSpinner > div > span {{
        color: #1e1e1e !important;
        text-shadow: 0 0 3px rgba(255,255,255,0.7);
        font-size: 0.84rem !important;
    }}

    /* ── Divider ── */
    hr {{
        border-color: rgba(90,130,90,0.35) !important;
        margin: 0.5rem 0 !important;
    }}

    /* ── Fade-in animation ── */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
""", unsafe_allow_html=True)

# ── Model config ──────────────────────────────────────────────────────────────
CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

tokenizer = tiktoken.get_encoding("gpt2")

@st.cache_resource
def load_model():
    model = GPTModel(BASE_CONFIG)
    checkpoint_path = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def format_input(entry_instruction, entry_input):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry_instruction}"
    )
    input_text = f"\n\n### Input:\n{entry_input}" if entry_input else ""
    return instruction_text + input_text

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Instruction-Tuned LLM Interface")
st.markdown(
    '<div class="subtitle">Interact with your fine-tuned <strong>GPT-2 Medium</strong> model using structured instructions.</div>',
    unsafe_allow_html=True,
)

# Instruction + Input side-by-side to save vertical space
col_inst, col_inp = st.columns(2)
with col_inst:
    instruction = st.text_area(
        "Instruction",
        "Summarize the following paragraph.",
        height=90,
    )
with col_inp:
    input_text = st.text_area(
        "Input (optional)",
        "Deep learning is a subset of machine learning...",
        height=90,
    )

# Advanced options — styled expander
with st.expander("⚙ Advanced options"):
    max_tokens = st.slider("Max tokens to generate", 10, 512, 128, step=1)

st.markdown("<div style='margin:0.3rem 0'></div>", unsafe_allow_html=True)

# Centred generate button
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    generate_clicked = st.button("Generate response")

# Response area — always visible, pre-allocated
st.markdown("<div style='margin:0.35rem 0 0.15rem'></div>", unsafe_allow_html=True)
response_container = st.empty()

if generate_clicked:
    with st.spinner("Thinking…"):
        formatted = format_input(instruction, input_text)
        input_ids = text_to_token_ids(formatted, tokenizer).to(device)
        token_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_tokens,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response = (
            generated_text[len(formatted):]
            .replace("### Response:", "")
            .strip()
        )

    response_container.markdown(
        f"""
        <div class="response-label">Generated response</div>
        <div class="response-box">{response}</div>
        """,
        unsafe_allow_html=True,
    )
else:
    response_container.markdown(
        """
        <div class="response-label">Generated response</div>
        <div class="response-box" style="color:#6a8a6a; font-style:italic;">
            Response will appear here…
        </div>
        """,
        unsafe_allow_html=True,
    )