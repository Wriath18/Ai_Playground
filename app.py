# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import time

# load_dotenv()

# # Environment variables
# groq_api_key = os.environ["GROQ_API_KEY"]

# # Available LLMs
# def create_llm(model_name, temperature):
#     return ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)

# llm_options = {
#     "Groq (llama-3.1-70b-versatile)": "llama-3.1-70b-versatile",
#     "Groq (llama-3.1-8b-instant)": "llama-3.1-8b-instant",
#     "Groq (llama3-70b-8192)": "llama3-70b-8192",
#     "Groq (llama3-8b-8192)": "llama3-8b-8192",
#     "Groq (gemma-7b-it)": "gemma-7b-it",
#     "Groq (gemma2-9b-it)": "gemma2-9b-it",
#     "Groq (mixtral-8x7b-32768)": "mixtral-8x7b-32768",
#     "Groq (llava-v1.5-7b-4096-preview)": "llava-v1.5-7b-4096-preview",
# }

# # Initialize chat prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     You are a technical and business consultant. Answer the following question to the best of your ability:
#     Question: {input}
#     """
# )

# def create_llm_window(window_number, col):
#     with col:
#         st.header(f"LLM {window_number}")
        
#         # LLM selection
#         selected_llm = st.selectbox(
#             f"Select LLM for Window {window_number}",
#             options=list(llm_options.keys()),
#             key=f"llm_select_{window_number}"
#         )
        
#         # Temperature slider
#         temperature = st.slider(
#             f"Temperature for LLM {window_number}",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.7,
#             step=0.1,
#             key=f"temperature_{window_number}"
#         )
        
#         # Initialize messages in session state if not present
#         if f'messages_{window_number}' not in st.session_state:
#             st.session_state[f'messages_{window_number}'] = []
        
#         # Display chat messages
#         for message in st.session_state[f'messages_{window_number}']:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
        
#         return selected_llm, temperature

# def process_query(prompt, llm, temp, window_number):
#     model_name = llm_options[llm]
#     llm_instance = create_llm(model_name, temp)
    
#     start_time = time.time()
#     response = llm_instance.invoke(prompt.format(input=prompt))
#     end_time = time.time()
    
#     response_time = end_time - start_time
    
#     full_response = f"{response.content}\n\n*Response time: {response_time:.2f} seconds*"
#     return full_response

# def main():
#     st.set_page_config(layout="wide")
#     st.title("AI Playground🤖 : LLM Comparison 💻💾")
#     st.markdown("This is a playground to compare different LLMs. Developed by [@Wriath18](https://github.com/Wriath18)")

    
#     # Shared input for both LLMs
#     user_input = st.chat_input("Enter your question for both LLMs")
    
#     # Create two columns for LLM windows
#     col1, col2 = st.columns(2)
    
#     # Assigning each window to a separate column
#     llm1, temp1 = create_llm_window(1, col1)
#     llm2, temp2 = create_llm_window(2, col2)
    
#     if user_input:
#         for i, (llm, temp, col) in enumerate([(llm1, temp1, col1), (llm2, temp2, col2)], start=1):
#             with col:
#                 st.session_state[f'messages_{i}'].append({"role": "user", "content": user_input})
#                 with st.chat_message("user"):
#                     st.markdown(user_input)

#                 with st.chat_message("assistant"):
#                     message_placeholder = st.empty()
#                     full_response = process_query(user_input, llm, temp, i)

#                     message_placeholder.markdown(full_response)
#                     st.session_state[f'messages_{i}'].append({"role": "assistant", "content": full_response})

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import time
# import random
# from nltk.translate.bleu_score import sentence_bleu
# from sklearn.metrics import f1_score

# load_dotenv()

# # Environment variables
# groq_api_key = os.environ["GROQ_API_KEY"]

# # Available LLMs
# def create_llm(model_name, temperature):
#     return ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)

# llm_options = {
#     "Groq (llama-3.1-70b-versatile)": "llama-3.1-70b-versatile",
#     "Groq (llama-3.1-8b-instant)": "llama-3.1-8b-instant",
#     "Groq (llama3-70b-8192)": "llama3-70b-8192",
#     "Groq (llama3-8b-8192)": "llama3-8b-8192",
#     "Groq (gemma-7b-it)": "gemma-7b-it",
#     "Groq (gemma2-9b-it)": "gemma2-9b-it",
#     "Groq (mixtral-8x7b-32768)": "mixtral-8x7b-32768",
#     "Groq (llava-v1.5-7b-4096-preview)": "llava-v1.5-7b-4096-preview",
# }

# # Initialize chat prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     You are a technical and business consultant. Answer the following question to the best of your ability:
#     Question: {input}
#     """
# )

# def create_llm_window(window_number, col):
#     with col:
#         st.header(f"LLM {window_number}")
        
#         # LLM selection
#         selected_llm = st.selectbox(
#             f"Select LLM for Window {window_number}",
#             options=list(llm_options.keys()),
#             key=f"llm_select_{window_number}"
#         )
        
#         # Temperature slider
#         temperature = st.slider(
#             f"Temperature for LLM {window_number}",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.7,
#             step=0.1,
#             key=f"temperature_{window_number}"
#         )
        
#         # Initialize messages in session state if not present
#         if f'messages_{window_number}' not in st.session_state:
#             st.session_state[f'messages_{window_number}'] = []
        
#         # Display chat messages
#         for message in st.session_state[f'messages_{window_number}']:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
        
#         return selected_llm, temperature

# def calculate_mock_f1_score():
#     # This is a mock function. In a real scenario, you'd compare the model's output to a gold standard.
#     return random.uniform(0.7, 1.0)

# def calculate_mock_bleu_score(response):
#     # This is a mock function. In a real scenario, you'd compare the model's output to reference translations.
#     reference = [response.split()]  # Using the response itself as a reference for demonstration
#     candidate = response.split()
#     return sentence_bleu(reference, candidate)

# def calculate_token_generation_rate(response, response_time):
#     # Estimating token count as words / 0.75 (rough estimate)
#     estimated_tokens = len(response.split()) / 0.75
#     return estimated_tokens / response_time

# def process_query(prompt, llm, temp, window_number):
#     model_name = llm_options[llm]
#     llm_instance = create_llm(model_name, temp)
    
#     start_time = time.time()
#     response = llm_instance.invoke(prompt.format(input=prompt))
#     end_time = time.time()
    
#     response_time = end_time - start_time
    
#     # Calculate benchmarks
#     f1_score = calculate_mock_f1_score()
#     bleu_score = calculate_mock_bleu_score(response.content)
#     token_gen_rate = calculate_token_generation_rate(response.content, response_time)
    
#     benchmarks = f"""
#     **Benchmarks:**
#     - F1 Score: {f1_score:.4f}
#     - BLEU Score: {bleu_score:.4f}
#     - Response Time: {response_time:.2f} seconds
#     - Token Generation Rate: {token_gen_rate:.2f} tokens/second
#     """
    
#     full_response = f"{response.content}\n\n{benchmarks}"
#     return full_response

# def main():
#     st.set_page_config(layout="wide")
#     st.title("AI Playground🤖 : LLM Comparison 💻💾")
#     st.markdown("This is a playground to compare different LLMs. Developed by [@Wriath18](https://github.com/Wriath18)")

    
#     # Shared input for both LLMs
#     user_input = st.chat_input("Enter your question for both LLMs")
    
#     # Create two columns for LLM windows
#     col1, col2 = st.columns(2)
    
#     # Assigning each window to a separate column
#     llm1, temp1 = create_llm_window(1, col1)
#     llm2, temp2 = create_llm_window(2, col2)
    
#     if user_input:
#         for i, (llm, temp, col) in enumerate([(llm1, temp1, col1), (llm2, temp2, col2)], start=1):
#             with col:
#                 st.session_state[f'messages_{i}'].append({"role": "user", "content": user_input})
#                 with st.chat_message("user"):
#                     st.markdown(user_input)

#                 with st.chat_message("assistant"):
#                     message_placeholder = st.empty()
#                     full_response = process_query(user_input, llm, temp, i)

#                     message_placeholder.markdown(full_response)
#                     st.session_state[f'messages_{i}'].append({"role": "assistant", "content": full_response})

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
import random
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score

load_dotenv()

# Environment variables
groq_api_key = os.environ["GROQ_API_KEY"]

# Available LLMs
def create_llm(model_name, temperature):
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)

llm_options = {
    "Groq (llama-3.1-70b-versatile)": "llama-3.1-70b-versatile",
    "Groq (llama-3.1-8b-instant)": "llama-3.1-8b-instant",
    "Groq (llama3-70b-8192)": "llama3-70b-8192",
    "Groq (llama3-8b-8192)": "llama3-8b-8192",
    "Groq (gemma-7b-it)": "gemma-7b-it",
    "Groq (gemma2-9b-it)": "gemma2-9b-it",
    "Groq (mixtral-8x7b-32768)": "mixtral-8x7b-32768",
    "Groq (llava-v1.5-7b-4096-preview)": "llava-v1.5-7b-4096-preview",
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
                    # Check if 'benchmarks' key exists
                    if "benchmarks" in message:
                        st.markdown(message["benchmarks"])
                        with st.expander("Click to view full response"):
                            st.markdown(message["content"])
                    else:
                        # For older messages without benchmarks
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])
        
        return selected_llm, temperature

def calculate_mock_f1_score():
    # This is a mock function. In a real scenario, you'd compare the model's output to a gold standard.
    return random.uniform(0.7, 1.0)

def calculate_mock_bleu_score(response):
    # This is a mock function. In a real scenario, you'd compare the model's output to reference translations.
    reference = [response.split()]  # Using the response itself as a reference for demonstration
    candidate = response.split()
    return sentence_bleu(reference, candidate)

def calculate_token_generation_rate(response, response_time):
    # Estimating token count as words / 0.75 (rough estimate)
    estimated_tokens = len(response.split()) / 0.75
    return estimated_tokens / response_time

def process_query(prompt, llm, temp, window_number):
    model_name = llm_options[llm]
    llm_instance = create_llm(model_name, temp)
    
    start_time = time.time()
    response = llm_instance.invoke(prompt.format(input=prompt))
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # Calculate benchmarks
    f1_score = calculate_mock_f1_score()
    bleu_score = calculate_mock_bleu_score(response.content)
    token_gen_rate = calculate_token_generation_rate(response.content, response_time)
    
    benchmarks = f"""
    **Benchmarks:**
    - F1 Score: {f1_score:.4f}
    - BLEU Score: {bleu_score:.4f}
    - Response Time: {response_time:.2f} seconds
    - Token Generation Rate: {token_gen_rate:.2f} tokens/second
    """
    
    return response.content, benchmarks

def main():
    st.set_page_config(layout="wide")
    st.title("AI Playground🤖 : LLM Comparison 💻💾")
    st.markdown("This is a playground to compare different LLMs. Developed by [@Wriath18](https://github.com/Wriath18)")

    
    # Shared input for both LLMs
    user_input = st.chat_input("Enter your question for both LLMs")
    
    # Create two columns for LLM windows
    col1, col2 = st.columns(2)
    
    # Assigning each window to a separate column
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
                    st.markdown(benchmarks)
                    with st.expander("Click to view full response"):
                        st.markdown(response_content)
                    st.session_state[f'messages_{i}'].append({
                        "role": "assistant", 
                        "content": response_content,
                        "benchmarks": benchmarks
                    })

if __name__ == "__main__":
    main()