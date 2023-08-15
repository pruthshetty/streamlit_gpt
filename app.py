import os
import sys
import time
import json
import openai
import requests
import tiktoken
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = "your-openai-api-key"
openai.api_key = "your-openai-api-key"

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def init_environment():
    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.load_local("faiss_index", embeddings)
    return faiss_db

def gen_user_input(docs, query) -> str:
    return {"role": "system", "content": docs}

def prompt(query):
    len_doc_1 = 512
    len_doc_2 = 256
    len_doc_3 = 128
    
    vectordb = init_environment()
    
    docs = vectordb.similarity_search(query, k=3)
    docs[0].page_content = truncate_string(docs[0].page_content, len_doc_1, "cl100k_base")
    docs[1].page_content = truncate_string(docs[1].page_content, len_doc_2, "cl100k_base")
    docs[2].page_content = truncate_string(docs[2].page_content, len_doc_3, "cl100k_base")
    docs = str(docs[0].page_content+" "+docs[1].page_content+" "+docs[2].page_content)
    prompt = gen_user_input(docs, query)
    return prompt

def tokenize_string(string: str, encoding_name: str) -> list:
    """Returns the tokens of a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)
    return tokens


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    tokens = tokenize_string(string, encoding_name)
    num_tokens = len(tokens)
    return num_tokens


def truncate_string(string: str, max_tokens: int, encoding_name: str) -> str:
    """Truncates a string to a maximum number of tokens."""

    # Tokenize the string
    tokens = tokenize_string(string, encoding_name)

    # If the string is already below the token limit, return it as is
    if len(tokens) <= max_tokens:
        return string

    # Otherwise, proceed with the truncation process...
    
    # Split the string into words
    words = string.split()

    # Initialize an empty list to hold the truncated words
    truncated_words = []

    # Initialize a counter for the number of tokens
    num_tokens = 0

    for word in words:
        # Tokenize the word
        word_tokens = tokenize_string(word, encoding_name)

        # If adding this word won't exceed the maximum number of tokens
        if num_tokens + len(word_tokens) <= max_tokens:
            # Add the word to the list of truncated words
            truncated_words.append(word)

            # Update the number of tokens
            num_tokens += len(word_tokens)
        else:
            # If adding this word would exceed the maximum number of tokens, stop truncating
            break

    # Join the truncated words back into a string
    truncated_string = ' '.join(truncated_words)

    return truncated_string

system_prompt = read_file("app_info/system_prompt.txt")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    system_prompt = read_file("app_info/system_prompt.txt")
    st.session_state.messages = [{"role": "system", "content": system_prompt}]


for message in st.session_state.messages:
    if message["role"] not in ["system"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

len_memory = 6 #Remembers last 2 messages in conversational context (rolling)
if user_input := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(prompt(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[st.session_state.messages[0]] + st.session_state.messages[-len_memory:],
            stream=True,
            max_tokens=512,
            temperature=0.1,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    pst_datetime = get_pst_now()
    record_id = write_to_airtable(user_input, str(pst_datetime))
