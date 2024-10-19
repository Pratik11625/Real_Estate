import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from shapely.geometry import Point
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables (like HuggingFace token)
load_dotenv()

# Title for the Streamlit App
st.title("Real Estate Property Search ")

# HuggingFace Token for API access
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    st.error("HuggingFace token not found! Please set HF_TOKEN in your .env file.")
    st.stop()

# Initialize the HuggingFace LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_token)

# Hugging Face Embedding Model for Document Retrieval
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Web-based document loader for real estate data
# URLs of real estate websites
urls = [
    "https://housing.com/in/buy/real-estate-pune",
    "https://www.magicbricks.com/property-for-sale-rent-in-Pune/residential-real-estate-Pune",
    # "https://www.99acres.com/search/property/buy/une?city=19&keyword=%7Bune&preference=S&area_unit=1&res_com=R",
    "https://www.squareyards.com/pune-real-estate",
    "https://www.makaan.com/pune-residential-property/buy-property-in-pune-city"
]

# Load documents from the web sequentially
data = []
for url in urls:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            page_content = soup.get_text()
            data.append(page_content)
        else:
            st.warning(f"Failed to retrieve data from {url}")
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")

if not data:
    st.error("No data loaded from any of the URLs.")
    st.stop()


# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_text("\n\n".join(data))


# Create FAISS VectorStore from the document splits
vectorstore = FAISS.from_texts(splits, embedding=embedding)

# Create a retriever for relevant snippets from the data
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Prompt Template for Real Estate Queries
promp_temp = """
You are an expert in real estate with 30 years of experience.
You know how to communicate with customers to find the best flat for them without leaving them empty-handed.
If we don't have the flat at a particular location as per their requirement, show them the best nearby choices with similar cost and size.
Answer the questions based on the context below.
If you can't answer, reply "I don't know".
Provide the data in bullet points.
Real estate property in a particular {location}, in {city}, {bhk}, {price}.
If any description {text}, include **builder name or group name**, 
"""

prompt = ChatPromptTemplate.from_messages([("system", promp_temp), ("user", "{text}")])

# Parser to handle output
parser = StrOutputParser()

# Collecting inputs from the user using Streamlit UI
# if st.sidebar == True:
bhk = st.sidebar.text_input("Provide the BHK:", value="2")
location_input = st.sidebar.text_input("Provide the location (state):", value="Maharashtra")
city = st.sidebar.text_input("Provide the city:", value="Pune")
price = st.sidebar.text_input("Provide the price range:", value="40 lakhs")
description = st.sidebar.text_input("Provide any specific description:", value="flat balcony facing east")

chain = prompt | llm | parser

st.sidebar.title("search your property...")

if st.sidebar.button("Search Property"):
    # Prepare the query text for the LLM
    query_text = f"Show me the property of {bhk} BHK in {city} city of {location_input} location, price starting from {price} with description {description}."
    try:
        # Retrieve relevant data from the document retriever
        response = retriever.get_relevant_documents(query_text)
        # st.success("Data retrieved from document retriever!")
        # st.success((response))
        # print(response)

        # Check if response is empty
        if not response:
            st.subheader("No results found in the retrieved data!")

        else:
            # Pass through the LLM and get results from the retriever data
            results = chain.invoke({
                "location": location_input,
                "city": city,
                "bhk": bhk,
                "price": price,
                "text": description
            },)

            if results:
                st.success("LLM Response:")
                st.success(results)
            else:
                st.subheader("No valid content found in the LLM results!")

    except Exception as e:
        st.error(f"Error fetching results: {e}")

else:
    st.error("Unable to find the location. Please try again.")
