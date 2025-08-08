import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextIteratorStreamer
import torch
import spacy
import warnings
from threading import Thread
import os

warnings.filterwarnings('ignore')

# GitHub model path (cloned automatically by Streamlit Cloud)
MODEL_REPO = "MarpakaPradeepSai/Chatbot"
MODEL_DIR = "Model/Final_Advanced_ticketing_events_DistilGPT2_fine-tuned"

# Clone the model repo if not already present
if not os.path.exists("Model"):
    os.system(f"git clone https://github.com/{MODEL_REPO}.git")

# Load model & tokenizer
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()

# Load SpaCy model for NER
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_trf")

nlp = load_spacy()

# Static placeholders dictionary
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}" : "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}" : "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    # ... (keep the rest of your placeholder mappings here)
}

# Extract dynamic placeholders
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

# Replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Generate response without streaming
def generate_response(instruction, max_length=256):
    device = model.device
    model.eval()
    dynamic_placeholders = extract_dynamic_placeholders(instruction)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    raw_response = response[response_start:].lstrip()
    return replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)

# Streamlit UI
st.title("🎟️ Event Ticketing Chatbot")
st.markdown("Ask me anything related to events, tickets, refunds, cancellations, etc.")

user_input = st.text_input("Enter your question:")
if st.button("Get Response"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            answer = generate_response(user_input)
        st.markdown(answer, unsafe_allow_html=True)
