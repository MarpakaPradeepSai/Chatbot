import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextIteratorStreamer
from threading import Thread
import torch
import time

# Configuration
MODEL_PATH = "MarpakaPradeepSai/Chatbot/main/Model"  # GitHub path to your model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_RESPONSE_LENGTH = 350
TEMPERATURE = 0.85
TOP_P = 0.92

# Page setup
st.set_page_config(
    page_title="Event Ticketing Assistant",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    }
    .chat-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }
    .user-message {
        background-color: #2575fc;
        color: white;
        border-radius: 15px 15px 0 15px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
    }
    .bot-message {
        background-color: #f1f2f6;
        color: #2f3542;
        border-radius: 15px 15px 15px 0;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: left;
    }
    .stTextInput>div>div>input {
        border-radius: 25px;
        padding: 15px;
    }
    .stButton>button {
        border-radius: 25px;
        padding: 10px 25px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a0db9 0%, #1c65e0 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model and tokenizer with progress indicators"""
    with st.spinner("🚀 Loading AI model... This may take a minute"):
        progress_bar = st.progress(0)
        
        # Update progress
        for i in range(10):
            progress_bar.progress((i + 1) * 10)
            time.sleep(0.2)
        
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        
        # Update progress to complete
        progress_bar.progress(100)
        time.sleep(0.5)
        
        return model.to(DEVICE), tokenizer

def generate_response(instruction, _model, _tokenizer):
    """Generate streaming response with the model"""
    input_text = f"Instruction: {instruction} Response:"
    inputs = _tokenizer(input_text, return_tensors='pt').to(DEVICE)
    
    streamer = TextIteratorStreamer(
        _tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True,
        timeout=20.0
    )
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_RESPONSE_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id
    )
    
    thread = Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return streamer

def main():
    """Main Streamlit application"""
    # Load model once
    model, tokenizer = load_model()
    
    # App header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: white; font-size: 3rem;">🎫 Event Ticketing Assistant</h1>
        <p style="color: white; font-size: 1.2rem;">Ask me about ticket sales, event details, or refund policies!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi there! I'm your event ticketing assistant. How can I help you today?"}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f'<div class="{"user-message" if message["role"] == "user" else "bot-message"}">'
                            f'{message["content"]}</div>', 
                            unsafe_allow_html=True)
        
        # User input
        if prompt := st.chat_input("Ask about tickets or events..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Generate response stream
                streamer = generate_response(prompt, model, tokenizer)
                
                # Display streaming response
                for chunk in streamer:
                    full_response += chunk
                    message_placeholder.markdown(
                        f'<div class="bot-message">{full_response} █</div>', 
                        unsafe_allow_html=True
                    )
                
                # Final message without cursor
                message_placeholder.markdown(
                    f'<div class="bot-message">{full_response}</div>', 
                    unsafe_allow_html=True
                )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
