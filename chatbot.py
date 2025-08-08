import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextIteratorStreamer
import torch
import spacy
import warnings
import time
from threading import Thread
import os
import requests

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Event Ticketing Chatbot",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #1976d2;
}
.bot-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
.stSpinner > div {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# GitHub directory containing the DistilGPT2 model files
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"

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

# Static placeholders dictionary
STATIC_PLACEHOLDERS = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}": "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{PAYMENT_SECTION}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}": "<b>Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>",
    "{{HELP_SECTION}}": "<b>Help</b>",
    "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>",
    "{{TICKET_DETAILS}}": "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}": "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}": "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
}

def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    """Download model files from GitHub repository."""
    os.makedirs(model_dir, exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(MODEL_FILES):
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        
        if not os.path.exists(local_path):
            status_text.text(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    with open(local_path, "wb") as f:
                        if total_size == 0:
                            f.write(response.content)
                        else:
                            downloaded = 0
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                    status_text.text(f"Downloaded {filename} ✓")
                else:
                    st.error(f"Failed to download {filename}. Status code: {response.status_code}")
                    return False, model_dir
            except requests.exceptions.RequestException as e:
                st.error(f"Network error downloading {filename}: {str(e)}")
                return False, model_dir
            except Exception as e:
                st.error(f"Error downloading {filename}: {str(e)}")
                return False, model_dir
        else:
            status_text.text(f"Found {filename} ✓")
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(MODEL_FILES))
    
    progress_bar.empty()
    status_text.empty()
    return True, model_dir

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer with caching for performance."""
    try:
        # Download model files from GitHub
        with st.spinner("Downloading model files from GitHub..."):
            success, model_dir = download_model_files()
            if not success:
                st.error("Failed to download model files. Please check your internet connection and try again.")
                st.stop()
        
        st.success("Model files downloaded successfully!")
        
        # Load tokenizer and model from the downloaded files
        with st.spinner("Loading model and tokenizer..."):
            tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
            model = GPT2LMHeadModel.from_pretrained(model_dir)
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        st.success("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check if all model files are available in the GitHub repository.")
        st.stop()

@st.cache_resource
def load_spacy_model():
    """Load SpaCy model with caching."""
    try:
        # Try to load the transformer model, fallback to smaller model if not available
        try:
            nlp = spacy.load("en_core_web_trf")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.error("SpaCy English model not found. Please install 'en_core_web_sm' or 'en_core_web_trf'")
                st.stop()
        return nlp
    except Exception as e:
        st.error(f"Error loading SpaCy model: {str(e)}")
        st.stop()

def extract_dynamic_placeholders(instruction, nlp):
    """Extract dynamic placeholders using SpaCy NER."""
    doc = nlp(instruction)
    dynamic_placeholders = {}

    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"

    # Default values if no entities are found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"

    return dynamic_placeholders

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    """Replace static and dynamic placeholders in the response."""
    # Replace static placeholders first
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)

    # Replace dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)

    return response

def generate_response_streaming(instruction, model, tokenizer, nlp, max_length=256):
    """Generate streaming response with real-time display."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    dynamic_placeholders = extract_dynamic_placeholders(instruction, nlp)

    # Build streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Prepare inputs
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)

    # Launch generation in a background thread
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

    # For Streamlit streaming
    response_placeholder = st.empty()
    raw_buffer = ""
    
    for new_text in streamer:
        raw_buffer += new_text
        processed = replace_placeholders(raw_buffer, dynamic_placeholders, STATIC_PLACEHOLDERS)
        processed = processed.lstrip()  # Remove leading whitespace
        
        # Update the display in real-time
        response_placeholder.markdown(f'<div class="bot-message">{processed}</div>', unsafe_allow_html=True)
        time.sleep(0.05)  # Small delay for visual effect

    thread.join()
    final_response = replace_placeholders(raw_buffer, dynamic_placeholders, STATIC_PLACEHOLDERS).lstrip()
    return final_response

def main():
    # Header
    st.markdown('<h1 class="main-header">🎫 Event Ticketing Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("Welcome to our AI-powered event ticketing assistant! Ask me anything about events, tickets, cancellations, refunds, and more.")

    # Load models
    with st.spinner("Loading AI models... This may take a moment on first run."):
        model, tokenizer = load_model_and_tokenizer()
        nlp = load_spacy_model()

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Example Questions")
        example_questions = [
            "How do I cancel my ticket?",
            "What is your refund policy?",
            "How can I upgrade my ticket?",
            "I need help with payment issues",
            "How do I transfer my ticket to someone else?",
            "Where can I find upcoming events?",
            "What are the cancellation fees?",
            "How do I contact customer support?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.user_input = question

    # Main chat interface
    st.subheader("💬 Chat with the Assistant")

    # Display chat history
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f'<div class="user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message"><strong>Assistant:</strong> {bot_msg}</div>', unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask your question:", 
                                 placeholder="Type your question about events, tickets, cancellations, etc.",
                                 value=st.session_state.get('user_input', ''))
        submitted = st.form_submit_button("Send")

    # Clear the temporary input value
    if 'user_input' in st.session_state:
        del st.session_state.user_input

    # Process user input
    if submitted and user_input:
        # Capitalize first letter
        formatted_input = user_input[0].upper() + user_input[1:] if user_input else ""
        
        # Display user message
        st.markdown(f'<div class="user-message"><strong>You:</strong> {formatted_input}</div>', unsafe_allow_html=True)
        
        # Generate and display bot response with streaming
        with st.spinner("Thinking..."):
            try:
                response = generate_response_streaming(formatted_input, model, tokenizer, nlp)
                
                # Add to chat history
                st.session_state.chat_history.append((formatted_input, response))
                
                # Limit chat history to last 10 exchanges to prevent memory issues
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Clear chat history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by GPT-2 | Built with Streamlit | Event Ticketing Assistant"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
