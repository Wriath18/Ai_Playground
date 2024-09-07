
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
import random
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score
import pandas as pd
import psutil  # For memory usage tracking

load_dotenv()

# Environment variables
groq_api_key = os.environ["GROQ_API_KEY"]

# Available LLMs
def create_llm(model_name, temperature):
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)

llm_options = {
    "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "llama3-70b-8192": "llama3-70b-8192",
    "llama3-8b-8192": "llama3-8b-8192",
    "gemma-7b-it": "gemma-7b-it",
    "gemma2-9b-it": "gemma2-9b-it",
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "llava-v1.5-7b-4096-preview": "llava-v1.5-7b-4096-preview",
}

# Initialize chat prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a technical and business consultant. Answer the following question to the best of your ability:
    Question: {input}
    """
)

def create_llm_window(window_number, col):
    with col:
        st.header(f"LLM {window_number}")
        
        # LLM selection
        selected_llm = st.selectbox(
            f"Select LLM for Window {window_number}",
            options=list(llm_options.keys()),
            key=f"llm_select_{window_number}"
        )
        
        # Temperature slider
        temperature = st.slider(
            f"Temperature for LLM {window_number}",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key=f"temperature_{window_number}"
        )
        
        # Initialize messages in session state if not present
        if f'messages_{window_number}' not in st.session_state:
            st.session_state[f'messages_{window_number}'] = []
        
        # Display chat messages
        for message in st.session_state[f'messages_{window_number}']:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if "benchmarks" in message:
                        st.markdown(message["benchmarks"])
                        with st.expander("Click to view full response"):
                            st.markdown(message["content"])
                    else:
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])
        
        return selected_llm, temperature

def calculate_mock_f1_score():
    return random.uniform(0.7, 1.0)

def calculate_mock_bleu_score(response):
    reference = [response.split()]  
    candidate = response.split()
    return sentence_bleu(reference, candidate)

def calculate_token_generation_rate(response, response_time):
    estimated_tokens = len(response.split()) / 0.75
    return estimated_tokens / response_time

def calculate_memory_usage():
    # Calculate memory usage in MB
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)
    return memory_usage

def calculate_mock_perplexity(response):
    # Perplexity is a placeholder here. In real scenarios, a model's perplexity can be calculated using
    # the probability distribution of predicted tokens.
    return random.uniform(10, 100)  # Mock value for perplexity

def process_query(prompt, llm, temp, window_number):
    model_name = llm_options[llm]
    llm_instance = create_llm(model_name, temp)
    
    start_time = time.time()
    start_memory = calculate_memory_usage()
    
    response = llm_instance.invoke(prompt.format(input=prompt))
    
    end_memory = calculate_memory_usage()
    end_time = time.time()
    
    response_time = end_time - start_time
    memory_diff = end_memory - start_memory
    
    # Additional benchmark calculations
    f1_score = calculate_mock_f1_score()
    bleu_score = calculate_mock_bleu_score(response.content)
    token_gen_rate = calculate_token_generation_rate(response.content, response_time)
    perplexity = calculate_mock_perplexity(response.content)
    
    # df = pd.DataFrame({
    #     "F1 Score": [f1_score],
    #     "BLEU Score": [bleu_score],
    #     "Response Time (seconds)": [response_time],
    #     "Token Generation Rate (tokens/second)": [token_gen_rate],
    #     "Memory Usage (MB)": [memory_diff],
    #     "Perplexity": [perplexity]
    # })

    df = pd.DataFrame({
        "Metric": ["F1 Score", "Response Time (seconds)", "Token Generation Rate (tokens/second)","Memory Usage (MB)","Perplexity"],
        "Value": [f1_score, response_time, token_gen_rate, memory_diff, perplexity]
    }).set_index("Metric")

    benchmarks = f"""
    **Benchmarks:**
    - F1 Score: {f1_score:.4f}
    - BLEU Score: {bleu_score:.4f}
    - Response Time: {response_time:.2f} seconds
    - Token Generation Rate: {token_gen_rate:.2f} tokens/second
    - Memory Usage: {memory_diff:.2f} MB
    - Perplexity: {perplexity:.2f}
    """
    
    return response.content, df

def main():
    st.set_page_config(layout="wide")
    st.title("AI Playground🤖 : LLM Comparison 💻💾")
    st.markdown("This is a playground to compare different LLMs. Developed by [@Wriath18](https://github.com/Wriath18)")

    user_input = st.chat_input("Enter your question for both LLMs")
    
    col1, col2 = st.columns(2)
    
    llm1, temp1 = create_llm_window(1, col1)
    llm2, temp2 = create_llm_window(2, col2)
    
    if user_input:
        for i, (llm, temp, col) in enumerate([(llm1, temp1, col1), (llm2, temp2, col2)], start=1):
            with col:
                st.session_state[f'messages_{i}'].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    response_content, benchmarks = process_query(user_input, llm, temp, i)
                    st.write(benchmarks)
                    with st.expander("Click to view full response"):
                        st.markdown(response_content)
                    st.session_state[f'messages_{i}'].append({
                        "role": "assistant", 
                        "content": response_content,
                        "benchmarks": benchmarks
                    })

if __name__ == "__main__":
    main()
