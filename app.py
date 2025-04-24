# ===============================
# Required Packages (pip install)
# ===============================
# pip install streamlit
# pip install ollama

import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# Load Ollama LLM locally (no API)
@st.cache_resource
def load_llm(model_name="llama3"):
    return Ollama(model=model_name)

# Function to create prompt
@st.cache_data
def create_prompt(text):
    return f"""
You are a topic classification expert.
Classify the following text into one of these categories: Finance, Legal, Technology, Healthcare, Education, Marketing, Science, Entertainment, or Other.

Text:
{text}

Respond with:
- Category: <category>
- Confidence (1-100): <confidence>
"""

# Streamlit app
st.title("🧠 Topic Classifier with Ollama")
st.markdown("Upload or paste your text to classify it into topics using a local LLM.")

model_name = st.sidebar.selectbox("Choose Ollama model:", ["llama3", "gemma", "deepseek-coder"])
llm = load_llm(model_name)

text_input = st.text_area("Paste text here:", height=200)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

text = text_input
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

if text:
    st.subheader("Classification Result")
    prompt = create_prompt(text)
    response = llm(prompt)
    st.text(response)

    # Parse result (basic parsing, customize if needed)
    if "Category:" in response:
        lines = response.strip().split("\n")
        cat_line = [l for l in lines if l.lower().startswith("category")]
        conf_line = [l for l in lines if "confidence" in l.lower()]
        
        if cat_line:
            st.success(cat_line[0])
        if conf_line:
            st.info(conf_line[0])
else:
    st.warning("Please provide text or upload a file to classify.")
