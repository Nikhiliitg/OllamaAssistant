from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Simple chatbot with Ollama"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

## Function to generate response
def generate_response(question, engine, temperature):
    llm = ollama.Ollama(model=engine, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Sidebar Inputs
engine = st.sidebar.selectbox("Select OLLAMA model", ["gemma:2b", "another_model"])
temperature = st.sidebar.selectbox("Select Temperature", [0.2, 0.5, 0.7, 1.0])
# Removed max_tokens as it's not supported by Ollama

## Main inference for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature)
    st.write("Assistant:", response)
else:
    st.write("Please provide user input")

