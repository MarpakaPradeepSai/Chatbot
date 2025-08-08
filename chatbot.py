import os
import io
import time
import zipfile
from threading import Thread
from typing import List, Dict

import streamlit as st
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="🎫 Ticketing Chatbot (DistilGPT-2)",
    page_icon="🎫",
    layout="wide",
)

st.markdown("""
    <style>
    .small-muted { color: #666; font-size: 0.85rem; }
    .param-label { font-weight: 600; }
    .stChatMessage [data-testid="stMarkdownContainer"] p { margin: 0.35rem 0; }
    .footer { color:#888; font-size:0.85rem; padding-top:0.5rem; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Constants and defaults
# -------------------------------
DEFAULT_MODEL_DIR = "Model"  # where the model will live locally
DEFAULT_GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Chatbot/tree/main/Model"

DEFAULTS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "trim_leading_ws": True,
    "use_history": False,
    "history_turns": 2,
    "seed": 42,
}

# -------------------------------
# Utils: device and GitHub download
# -------------------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _parse_github_tree_url(tree_url: str):
    # Expecting: https://github.com/<user>/<repo>/tree/<branch>/<subdir...>
    from urllib.parse import urlparse
    parts = urlparse(tree_url)
    path_parts = [p for p in parts.path.split('/') if p]
    if len(path_parts) < 5 or path_parts[2] != "tree":
        raise ValueError("Provide a GitHub 'tree' URL like: https://github.com/<user>/<repo>/tree/<branch>/<subdir>")
    user, repo, _, branch = path_parts[:4]
    sub_path = "/".join(path_parts[4:])
    return user, repo, branch, sub_path

def download_github_subdir(tree_url: str, dest_dir: str, force: bool = False):
    """
    Download a subfolder from a public GitHub repo by fetching the repo ZIP and
    extracting only that subfolder into dest_dir.
    """
    user, repo, branch, sub_path = _parse_github_tree_url(tree_url)
    zip_url = f"https://codeload.github.com/{user}/{repo}/zip/refs/heads/{branch}"

    if os.path.isdir(dest_dir) and os.listdir(dest_dir) and not force:
        return dest_dir

    # Fetch zip in-memory
    with st.spinner("Downloading model from GitHub…"):
        r = requests.get(zip_url, timeout=180)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))

    repo_prefix = f"{repo}-{branch}/"
    members = [m for m in z.namelist() if m.startswith(repo_prefix + sub_path + "/")]
    if not members:
        raise FileNotFoundError(f"Could not find subfolder '{sub_path}' in the GitHub repo archive.")

    # Extract only the desired subdir
    os.makedirs(dest_dir, exist_ok=True)
    for member in members:
        rel = member[len(repo_prefix + sub_path + "/"):]
        if not rel:
            continue
        target_path = os.path.join(dest_dir, rel)
        if member.endswith("/"):
            os.makedirs(target_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with z.open(member) as src, open(target_path, "wb") as out:
                out.write(src.read())

    return dest_dir

@st.cache_resource(show_spinner=False)
def ensure_model_local(github_tree_url: str, local_dir: str):
    # If it's already there, skip download
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir
    return download_github_subdir(github_tree_url, local_dir, force=True)

# -------------------------------
# Model loading and generation
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32  # CPU-friendly
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer

def build_prompt(messages: List[Dict], use_history: bool, history_turns: int, current_user_text: str) -> str:
    parts = []
    if use_history:
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
        for u, a in zip(user_msgs[-history_turns:], assistant_msgs[-history_turns:]):
            parts.append(f"Instruction: {u}\nResponse: {a}")
    parts.append(f"Instruction: {current_user_text}\nResponse:")
    return "\n\n".join(parts)

def stream_generate(model, tokenizer, device, prompt: str, gen_params: dict, trim_leading_ws: bool, container):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        timeout=120.0,
        decode_kwargs={"skip_special_tokens": True}
    )

    generator = None
    if gen_params.get("seed") is not None:
        generator = torch.Generator(device=device).manual_seed(int(gen_params["seed"]))

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        repetition_penalty=gen_params["repetition_penalty"],
        max_new_tokens=gen_params["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        generator=generator,
    )

    holder = {"sequences": None}
    def _run():
        out = model.generate(**generation_kwargs)
        holder["sequences"] = out

    t0 = time.time()
    thread = Thread(target=_run, daemon=True)
    thread.start()

    text_area = container.empty()
    streamed_text = ""
    started = not trim_leading_ws

    for new_text in streamer:
        if not started:
            new_text = new_text.lstrip()
            if not new_text:
                continue
            started = True
        streamed_text += new_text
        text_area.markdown(streamed_text)

    thread.join()
    t1 = time.time()

    total_tokens = 0
    try:
        sequences = holder["sequences"]
        total_tokens = int(sequences.shape[1] - input_ids.shape[1])
    except Exception:
        pass

    gen_time = max(t1 - t0, 1e-6)
    tps = total_tokens / gen_time if total_tokens > 0 else 0.0

    return streamed_text, {
        "time_s": gen_time,
        "new_tokens": total_tokens,
        "tokens_per_s": tps
    }

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Configure model source and decoding params")

    github_url = st.text_input(
        "GitHub model folder URL",
        value=DEFAULT_GITHUB_MODEL_URL,
        help="Public GitHub URL to the folder with model files (e.g., config.json, tokenizer files, weights)."
    )
    model_dir = st.text_input(
        "Local model directory",
        value=DEFAULT_MODEL_DIR,
        help="Where the model will be stored locally in the Streamlit container."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        temperature = st.slider("Temperature", 0.0, 1.5, DEFAULTS["temperature"], 0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, DEFAULTS["top_p"], 0.05)
    with col_b:
        repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, DEFAULTS["repetition_penalty"], 0.01)
        max_new_tokens = st.slider("Max new tokens", 16, 1024, DEFAULTS["max_new_tokens"], 8)

    st.markdown("---")
    use_history = st.checkbox("Use recent chat history as context", value=DEFAULTS["use_history"])
    history_turns = st.slider("History turns", 1, 6, DEFAULTS["history_turns"], disabled=not use_history)

    st.markdown("---")
    trim_leading_ws = st.checkbox("Trim leading whitespace on stream", value=DEFAULTS["trim_leading_ws"])
    set_seed = st.checkbox("Set random seed", value=True)
    seed = st.number_input("Seed", min_value=0, max_value=2**31-1, value=DEFAULTS["seed"], step=1, disabled=not set_seed)

    st.markdown("---")
    if st.button("♻️ Reload model"):
        ensure_model_local.clear()
        load_model.clear()
        st.success("Model cache cleared. It will reload on next run.")

# -------------------------------
# Ensure model is available (download if needed), then load
# -------------------------------
device = get_device()

if not os.path.isdir(model_dir) or not os.listdir(model_dir):
    try:
        ensure_model_local(github_url, model_dir)
    except Exception as e:
        st.error(f"Failed to fetch model from GitHub:\n{e}")
        st.stop()

with st.spinner("Loading model..."):
    model, tokenizer = load_model(model_dir, device)

# -------------------------------
# Chat UI
# -------------------------------
st.title("🎫 Ticketing Chatbot (DistilGPT‑2, Streamlit)")
st.caption("Ask about selling/exchanging tickets. Responses stream live without the leading gap.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How can I help you with your event tickets today?"})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Type your question...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        gen_params = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "seed": int(seed) if set_seed else None
        }

        prompt = build_prompt(
            messages=st.session_state.messages,
            use_history=use_history,
            history_turns=history_turns,
            current_user_text=user_text
        )

        final_text, stats = stream_generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            gen_params=gen_params,
            trim_leading_ws=trim_leading_ws,
            container=st.container()
        )

        st.session_state.messages.append({"role": "assistant", "content": final_text})
        st.markdown(
            f"<div class='small-muted'>⏱ {stats['time_s']:.2f}s • 🧠 {stats['new_tokens']} tokens • ⚡ {stats['tokens_per_s']:.1f} tok/s</div>",
            unsafe_allow_html=True
        )

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("🔁 Regenerate last"):
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "assistant":
                st.session_state.messages.pop(i)
                break
        st.rerun()
with col3:
    if st.download_button(
        "💾 Download chat",
        data="\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]),
        file_name="chat_transcript.txt",
        mime="text/plain",
    ):
        pass

st.markdown("<div class='footer'>Tip: if download from GitHub is slow or fails due to LFS limits, consider adding the Model folder to this repo or hosting on the Hugging Face Hub.</div>", unsafe_allow_html=True)
