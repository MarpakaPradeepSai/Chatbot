import os
import requests
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextIteratorStreamer
import torch
import spacy
import warnings
from threading import Thread

warnings.filterwarnings("ignore")

# --------------------
# GitHub model details
# --------------------
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Chatbot/raw/main/Model"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]

def download_model_files(model_dir="/tmp/Chatbot_Model"):
    """Download model files from GitHub if not already present."""
    os.makedirs(model_dir, exist_ok=True)
    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            r = requests.get(url)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
            else:
                st.error(f"❌ Failed to download {filename} from GitHub.")
                return None
    return model_dir

@st.cache_resource
def load_model_and_tokenizer():
    """Download and load the GPT-2 model."""
    model_dir = download_model_files()
    if not model_dir:
        st.stop()
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

model, tokenizer = load_model_and_tokenizer()
nlp = load_spacy_model()

# --------------------
# Placeholders
# --------------------
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}" : "www.support-team.com",
    "{{CITY}}": "city",
    "{{EVENT}}": "event"
    # Add rest from your original dictionary...
}

# --------------------
# Helper functions
# --------------------
def extract_dynamic_placeholders(instruction):
    doc = nlp(instruction)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":  
            dynamic_placeholders['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            dynamic_placeholders['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def generate_response_streaming(instruction, max_length=256):
    device = model.device
    model.eval()
    dynamic_placeholders = extract_dynamic_placeholders(instruction)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)

    gen_kwargs = dict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    final_text = ""
    for new_text in streamer:
        final_text += new_text
        processed = replace_placeholders(final_text, dynamic_placeholders, static_placeholders)
        yield processed

    thread.join()

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Event Ticketing Chatbot", layout="wide")
st.title("🎟️ Event Ticketing Chatbot")

user_input = st.text_input("Ask me something about events or tickets:")

if user_input:
    st.write("**Chatbot:**")
    placeholder = st.empty()
    streamed_text = ""
    for partial_output in generate_response_streaming(user_input):
        streamed_text = partial_output
        placeholder.markdown(streamed_text, unsafe_allow_html=True)
