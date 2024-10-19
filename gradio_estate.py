#lib
import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()

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
            print(f"Failed to retrieve data from {url}")
    except Exception as e:
        print(f"Error loading data from {url}: {e}")

if not data:
    print("No data loaded from any of the URLs.")
    # No need to stop the app, but raise an alert


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

If we don't have the flat at a particular location as per their requirement, show them the best nearby choices with similar cost and size and
also show them in the below formate.
Answer the questions based on the context below.

If you can't answer, reply "I don't know" or visit the links provided, show them from were you have gather the data.

show all the property, minimun=2 maximum=5.
Provide the data in bullet points, with sub-headings as follows:
- \nBuilder/Group Name: [Insert builder name or group name from the website of the property]
- Location: {location}, in {city}
- BHK Configuration: {bhk}
- Price: {price}
- Description: 
    - {text}
    - Provide a brief overview of the property features or additional details about the flat, neighborhood, or nearby facilities.
"""


prompt = ChatPromptTemplate.from_messages([("system", promp_temp), ("user", "{text}")])

# Parser to handle output
parser = StrOutputParser()

chain = prompt | llm | parser

# Function to process user's property search query
def response(bhk, location_input, city, price, description):
    query_text = f"Show me the property of {bhk} BHK in {city} city of {location_input} location, price starting from {price} with description {description}."

    try:
        # Simulate the retriever for relevant documents based on the query
        relevant_docs = [doc for doc in data if location_input.lower() in doc.lower()]
        response = retriever.get_relevant_documents(query_text)
        print(response)
        
        if not response:
            return "No results found for the given location."
        else:
            # Simulate LLM result based on the document retriever data
            # Assuming `chain.invoke()` passes through some LLM model for property recommendations
            # Simulated response based on query
            # results = f"Properties found: \n- {bhk} in {city}, {location_input}.\n- Price: {price}\n- Description: {description}\n- Based on the relevant real estate sources."
            results = chain.invoke({
                "location": location_input,
                "city": city,
                "bhk": bhk,
                "price": price,
                "text": description
            },)

            return results

    except Exception as e:
        return f"Error fetching results: {e}"


# Define the Gradio interface
demo = gr.Interface(
    fn=response,
    inputs=[
        gr.Radio(["1BHK", "2BHK", "2.5BHK", "3BHK", "4BHK"], value="2BHK", label="BHK", info="Choose your suitable BHK"),
        gr.Textbox(lines=1, label='Location', placeholder="e.g. Maharashtra"),
        gr.Textbox(lines=1, label='City', placeholder="e.g. Pune"),
        gr.Slider(2500000, 40000000, value=3500000, label="Price", step=100000, info="Choose the price starting from[lkh - cr]"),
        gr.Textbox(lines=2, label='Description', placeholder="e.g. East facing flat, near main road, nearby facilities"),
    ],
    outputs="text",
    title="Real Estate Property Search",
    description="This is a simple real estate property search app based on your inputs.",
    examples=[
        ["2BHK", "Maharashtra", "Pune", 3500000, "East facing"],
        ["2.5BHK", "Maharashtra", "Nagpur", 6000000, "North facing"],
        ["3BHK", "Maharashtra", "Mumbai", 25000000, "nearby facilities"],
    ]
)

# Launch the Gradio app
# demo.launch(share=True)
demo.launch()