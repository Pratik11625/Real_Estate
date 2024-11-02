
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
from langchain_ollama import ChatOllama
# Load environment variables (like HuggingFace token)
load_dotenv()

# Title for the Streamlit App
st.title("Real Estate Property Search ")

# HuggingFace Token for API access
hf_token = os.getenv('HF_TOKEN')
# os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
if not hf_token:
    # Set up HuggingFace embeddings model using an environment variable for the API token
      # Use '=' for assignment
    st.error("HuggingFace token not found! Please set HF_TOKEN in your .env file.")
    st.stop()

# Initialize the HuggingFace LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_token)
# llm = ChatOllama(model="qwen2.5:7b")
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
# promp_temp = """
# You are an expert in real estate with 30 years of experience.
# You know how to communicate with customers to find the best flat for them without leaving them empty-handed.

# If we don't have the flat at a particular location as per their requirement, show them the best nearby choices with similar cost and bhk and
# make sure that the near by  are 10 KM from the Pune City,also show them in the below formate.
# Answer the questions based on the context below.

# If you can't answer, reply "I don't know" or visit the links provided, show them from were you have gather the data.

# atleat you should show more option to select the property.
# Provide the data in bullet points, with sub-headings as follows:
# - \n**Builder/Group Name**: [Insert builder name or group name from the website of the property]
# - **Location**: {location}, in {city}
# - **BHK Configuration**: {bhk}
# - **Property_Type**: {property_type}
# - **Price**: {price}
# - **Description**: 
#     - {text}{amenities}{furnishing_status}{property_purpose}
#     - Provide a brief overview of the property features or additional details about the flat, neighborhood, or nearby facilities.\n\n  
# """

# promp_temp = """
# You are an expert in real estate with 30 years of experience in helping customers find their dream properties. Your goal is to provide detailed, structured, and helpful suggestions for properties based on the user's requirements.

# Use the following format to present information:
# - If a property matching all criteria is found, present it in detail.
# - If a direct match isn't available, suggest nearby or similar properties that may fulfill the user's needs.

# Provide the property information in bullet points using the structure below:
# - **Builder/Group Name**: [Insert builder or group name from the property listing]
#     - **Location**: [Location of the property], in [City]
#     - **BHK Configuration**: [BHK configuration of the property]
#     - **Price**: [Price of the property]
#     - **Description**:
#         - [Provide a brief overview of the property, including its features, nearby facilities, and other points of interest]

# If you cannot find a suitable property, respond with: "I couldn't find an exact match, but here are some alternatives nearby."

# The user requirements are:
# - BHK Configuration: {bhk}
# - Location: {location}, in {city}
# - Price Range: {price_range}
# - Specific Requirements: {description}

# Now, provide a detailed response based on the above requirements and available context.
# """

# promp_temp = """
# You are an expert in real estate with 30 years of experience.
# You know how to communicate with customers to find the best flat for them without leaving them empty-handed.

# If we don't have the flat at a particular location as per their requirement, show them the best nearby choices with similar cost and BHK and
# make sure that the nearby properties are within 10 KM from the Pune City. Show the options in the format below.

# If you can't answer, reply "I don't know" or visit the links provided, show them where you have gathered the data.

# You should show at least more than one option for property selection.
# Provide the data in bullet points, with sub-headings as follows:
# - **Builder/Group Name**: [Insert builder name or group name from the website of the property]
# - **Location**: {location}, in {city}
# - **BHK Configuration**: {bhk}
# - **Property_Type**: {property_type}
# - **Price**: {price}
# - **Furnishing Status**: {furnishing_status}
# - **Purpose**: {property_purpose}
# - **Amenities**: {amenities}
# - **Description**: 
#     - {text}
#     - Provide a brief overview of the property features or additional details about the flat, neighborhood, or nearby facilities.
# """

promp_temp= """
You are an expert in real estate with 30 years of experience, skilled at helping customers find the perfect property. If a flat is not available at the requested location or price, suggest similar options nearby, with similar cost and size. Answer the queries based on the context below, and provide all information in bullet points.

- 🏠 :blue[**Property 1**]
- 🏠 :blue[**Property 2**]
- 🏠 :blue[**Property 3**]
- 🏠 :blue[**Property 4**]
- 🏠 :blue[**Property 5**] 

Each property should include the following details, if available:
1.  - **Builder/Group Name**: [Insert builder name or group name from the website of the property]
# - **Location**: {location}, in {city}
# - **BHK Configuration**: {bhk}
# - **Price**: {price}
# - **description**: {description}
#     - Provide a brief overview of the property features or additional details about the flat, neighborhood, or nearby facilities.
4. **Project Amenities**: Provide a brief description of key features, amenities,  additional details about the flat, neighborhood, or nearby facilities. and any unique selling points of the property.
5. **Contact Details**: If there is a contact seller or contact owner option, include that. Otherwise, do not show it..
In case the builder_name, by_group_name is provided show them, else don't show

### In case no exact match is found:
If a flat is not available in the requested location or price range, show the message:
**"We've found similar properties for you based on your input."**
Then, list 5 similar properties based on BHK configuration and price range.

"""

prompt = ChatPromptTemplate.from_messages([("system", promp_temp),])

# Parser to handle output
parser = StrOutputParser()

# Collecting inputs from the user using Streamlit UI
# if st.sidebar == True:
# st.sidebar.title("Search your :blue[Property] :search:")
# bhk = st.sidebar.selectbox("Provide the BHK:",("1 BHK", "2 BHK", "2.5 BHK", "3 BHK", "4 BHK",) )
# location_input = st.sidebar.text_input("Provide the location (state):", value="Maharashtra")
# city = st.sidebar.text_input("Provide the city:", value="Pune")
# price = st.sidebar.slider("Provide the price range [in lakhs]:",25.50, 400.50 ,(40,60) )
# description = st.sidebar.text_input("Provide any specific description:", value="flat balcony facing east")

# Sidebar title with search icon
st.sidebar.title("🏠 Search Your Property... 🔍")

# BHK dropdown
bhk = st.sidebar.selectbox("Select BHK Configuration:", 
                           ("1 BHK", "2 BHK", "2.5 BHK", "3 BHK", "4 BHK", "5 BHK"),
                           index=2,
        help="Select the number of bedrooms, hall, and kitchen configuration you need.")

# # Property type dropdown (Added for broader search)
# property_type = st.sidebar.selectbox("Select Property Type:", 
#                                      ("Apartment", "Flat", "Bungalow", "House", "Villa"))

# Location input fields
location_input = st.sidebar.text_input("Enter State:", value="Maharashtra",help="Select the state in which you are looking for the property.")
city = st.sidebar.text_input("Enter Location:", value="Pune",help="Enter the name of the city where you want to search for properties.")

# Price range slider
# price = st.sidebar.slider("Set Your Budget (in Lakhs):", 
#                           min_value=25.0, max_value=400.0, value=(40.0, 60.0), step=5.0)
price =st.sidebar.slider(
        "Select Price Range (in lakhs):",
        min_value=10,
        max_value=500,
        value=(40, 80),
        step=5,
        help="Select the minimum and maximum price range for the property in lakhs."
)

# floor_preference = st.sidebar.selectbox(
#         "Select Floor Preference:",
#         options=["Ground Floor", "1st to 3rd Floor", "Above 3rd Floor", "Penthouse"],
#         help="Select your preferred floor level for the flat."
#     )

# New dropdown for property purpose (buy or rent)
# property_purpose = st.sidebar.selectbox("Select Purpose:", ("Buy", "Rent"))

# Dropdown for furnishing status (Added to provide more search options)
# furnishing_status = st.sidebar.selectbox("Furnishing Status:", 
#                                          ("Fully Furnished", "Semi-Furnished", "Unfurnished"))

# Additional search fields for user preferences
description = st.sidebar.text_area("Enter Specific Requirements:", 
                                #    value="Flat with east-facing balcony, near metro station.",
                                value="Amenities",
                                height=100,
        help="Enter any specific requirements you have, such as amenities, property orientation, or nearby facilities."
    )

# Amenities dropdown (optional search filter)
# amenities = st.sidebar.multiselect("Select Desired Amenities:",
#                                    ["Swimming Pool", "Gym", "Play Area", "Parking", "Security", "Clubhouse"])


chain = prompt | llm | parser

st.sidebar.title("search your property...")

if st.sidebar.button("🔍 Search Property"):
    # Prepare the query text for the LLM
    # query_text = f"Show me the property of {bhk} BHK in {city} city of {location_input} location, price starting from {price} with description {description}."
    query_text = f"Show me the property of {bhk} in {city} city of {location_input} location, price starting from {price}Lakhs with provided description {description}."

    try:
        # Retrieve relevant data from the document retriever
        response = retriever.get_relevant_documents(query_text)
        # st.success("Data retrieved from document retriever!")
        # st.success((response[1].page_content))
        # print(response)
        st.info(query_text)

        # Check if response is empty
        if not response:
            st.subheader("No results found in the retrieved data!")

        else:
            # Pass through the LLM and get results from the retriever data
            results = chain.invoke({
                # "location": location_input,
                # "city": city,
                # "property_type":property_type,
                # "bhk": bhk,
                # "price": price,
                # "text": description,
                # "amenities": amenities,
                "location": location_input,
                "city": city,
                # "property_type": property_type,
                "bhk": bhk,
                "price": price,
                "description": description,
                # "furnishing_status": furnishing_status,
                # "property_purpose": property_purpose,
                # "amenities": amenities,

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
    st.info("Please provide details and click 'Search Property' to find available properties.")
