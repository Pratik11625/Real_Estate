import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load environment variables (like HuggingFace token)
load_dotenv()

# Title for the Streamlit App
st.title("Real Estate Property Search")

# HuggingFace Token for API access
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Initialize the HuggingFace LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=os.getenv("HF_TOKEN"))


# Prompt Template for Real Estate Queries
promp_temp = """You are an expert in real estate with 30 years of experience.
You know how to communicate with customers to find the best flat for them without leaving them empty-handed.
If we don't have the flat at a particular location as per their requirement, show them the best nearby choices with similar cost and size.
Answer the questions based on the context below.
If you can't answer, reply "I don't know".
.real estate property in a particular {location}, in {city}, {bhk}, {price},
 [ latitude,longitude ] of the property, """

prompt = ChatPromptTemplate.from_messages([("system", promp_temp), ("user", "{text}")])

# Parser to handle output
parser = StrOutputParser()

chain = prompt | llm | parser

# Collecting inputs from the user using Streamlit UI
bhk = st.text_input("Provide the BHK:", value="2")
location = st.text_input("Provide the location (state):", value="Maharashtra")
city = st.text_input("Provide the city:", value="Pune")
price = st.text_input("Provide the price range:", value="40 lakhs")

# Query button
if st.button("Search Property"):
    # Run the query against the FAISS retriever and agent
    query_text = f"Show me the property of {bhk}BHK in {city} city of {location} location, price starting from {price}."
    results = chain.invoke({
        "location": location,
        "city": city,
        "bhk": bhk,
        "price": price,
        "text": query_text
    })

    # Display the result in the Streamlit app
    if results:
        st.subheader("Property Search Results")
        st.success(results)
    else:
        st.subheader("No results found!")
