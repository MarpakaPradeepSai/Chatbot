import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.generation.streamers import TextIteratorStreamer
import requests
import os
import spacy
import time
from threading import Thread

# GitHub directory containing the DistilGPT2 model files (your current model)
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Chatbot/raw/main/Model"

# List of model files to download
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    os.makedirs(model_dir, exist_ok=True)

    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
            else:
                st.error(f"Failed to download {filename} from GitHub.")
                return False
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # Ensure PAD token is set for GPT-2 style models
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

# Define static placeholders
static_placeholders = {
    "{{APP}}": "<b>App</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>",
    "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{WEBSITE_URL}}": "www.events-ticketing.com"
}

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

# Non-streaming fallback (kept for similarity/reference)
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# Streaming generator using TextIteratorStreamer (no leading gap)
def generate_response_stream(model, tokenizer, instruction, message_placeholder,
                             dynamic_placeholders, static_placeholders,
                             max_new_tokens=256, temperature=0.7, top_p=0.95):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Streamer setup
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        timeout=120.0,
        decode_kwargs={"skip_special_tokens": True}
    )

    # Background generation
    def _generate():
        with torch.no_grad():
            model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

    thread = Thread(target=_generate, daemon=True)
    thread.start()

    # Live stream to UI with leading whitespace trimmed
    started = False
    buffer_text = ""

    for new_text in streamer:
        if not started:
            new_text = new_text.lstrip()
            if not new_text:
                continue
            started = True

        buffer_text += new_text
        display_text = replace_placeholders(buffer_text, dynamic_placeholders, static_placeholders)
        message_placeholder.markdown(display_text, unsafe_allow_html=True)

    thread.join()
    final_text = replace_placeholders(buffer_text, dynamic_placeholders, static_placeholders)
    return final_text

# CSS styling
st.markdown(
    """
<style>
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1.2em;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 5px;
    width: auto;
    min-width: 100px;
    font-family: 'Times New Roman', Times, serif !important;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
    color: white !important;
}
.stButton>button:active {
    transform: scale(0.98);
}
* { font-family: 'Times New Roman', Times, serif !important; }
.stSelectbox > div > div > div > div { font-family: 'Times New Roman', Times, serif !important; }
.stTextInput > div > div > input { font-family: 'Times New Roman', Times, serif !important; }
.stTextArea > div > div > textarea { font-family: 'Times New Roman', Times, serif !important; }
.stChatMessage { font-family: 'Times New Roman', Times, serif !important; }
.st-emotion-cache-r421ms { font-family: 'Times New Roman', Times, serif !important; }
.streamlit-expanderContent { font-family: 'Times New Roman', Times, serif !important; }
</style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for the "Ask this question" button
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6);
    color: white !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for horizontal line separator
st.markdown(
    """
<style>
    .horizontal-line {
        border-top: 2px solid #e0e0e0;
        margin: 15px 0;
    }
</style>
    """,
    unsafe_allow_html=True,
)

# --- New CSS for Chat Input Shadow Effect ---
st.markdown(
    """
<style>
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state for controlling disclaimer visibility and model loading status
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# Example queries for dropdown
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?",
    "How can I sell my ticket?"
]

# First, display loading message and load models
if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources... Please wait..."):
        try:
            # Initialize spaCy model for NER
            nlp = load_spacy_model()

            # Load DistilGPT2 model and tokenizer
            model, tokenizer = load_model_and_tokenizer()

            if model is not None and tokenizer is not None:
                st.session_state.models_loaded = True
                st.session_state.nlp = nlp
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
            else:
                st.error("Failed to load the model. Please refresh the page and try again.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# Display Disclaimer and Continue button only after models are loaded
if st.session_state.models_loaded and not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: Arial, sans-serif;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center;">⚠️Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents, and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24;">
                <li>Cancel Ticket</li>
                <li>Buy Ticket</li>
                <li>Sell Ticket</li>
                <li>Transfer Ticket</li>
                <li>Upgrade Ticket</li>
                <li>Find Ticket</li>
                <li>Change Personal Details on Ticket</li>
                <li>Get Refund</li>
                <li>Find Upcoming Events</li>
                <li>Customer Service</li>
                <li>Check Cancellation Fee</li>
                <li>Track Cancellation</li>
                <li>Ticket Information</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents.
                Even if the model fails to provide accurate responses from the predefined intents, we kindly ask for your patience and encourage you to try again.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right using columns
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun()

# Show chat interface only after clicking Continue and models are loaded
if st.session_state.models_loaded and st.session_state.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # Dropdown and Button section at the TOP, before chat history and input
    selected_query = st.selectbox(
        "Choose a query from examples:",
        ["Choose your question"] + example_queries,
        key="query_selectbox",
        label_visibility="collapsed"
    )
    process_query_button = st.button("Ask this question", key="query_button")

    # Access loaded models from session state
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_role = None  # Track last message role

    # Display chat messages from history
    for message in st.session_state.chat_history:
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    # Process selected query from dropdown
    if process_query_button:
        if selected_query == "Choose your question":
            st.error("⚠️ Please select your question from the dropdown.")
        elif selected_query:
            prompt_from_dropdown = selected_query
            prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

            st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
            if last_role == "assistant":
                st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt_from_dropdown, unsafe_allow_html=True)
            last_role = "user"

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                generating_response_text = "Generating response..."
                with st.spinner(generating_response_text):
                    dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp)
                    # Streamed generation (no leading whitespace gap)
                    full_response = generate_response_stream(
                        model=model,
                        tokenizer=tokenizer,
                        instruction=prompt_from_dropdown,
                        message_placeholder=message_placeholder,
                        dynamic_placeholders=dynamic_placeholders,
                        static_placeholders=static_placeholders,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.95
                    )

                # Ensure final content is displayed (already streamed, but safe to set)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"

    # Input box at the bottom
    if prompt := st.chat_input("Enter your own question:"):
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
        if not prompt.strip():
            st.toast("⚠️ Please enter a question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
            if last_role == "assistant":
                st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt, unsafe_allow_html=True)
            last_role = "user"

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                generating_response_text = "Generating response..."
                with st.spinner(generating_response_text):
                    dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                    # Streamed generation (no leading whitespace gap)
                    full_response = generate_response_stream(
                        model=model,
                        tokenizer=tokenizer,
                        instruction=prompt,
                        message_placeholder=message_placeholder,
                        dynamic_placeholders=dynamic_placeholders,
                        static_placeholders=static_placeholders,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.95
                    )

                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"

    # Conditionally display reset button
    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            last_role = None
            st.rerun()
