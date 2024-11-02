import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv

# Load environment variables (like HuggingFace token and API key)
load_dotenv()

# Title for the Streamlit App
st.title("🏠 :blue[Real ] Estate :red[Property] Search with Chat History")
# st.markdown(" Using  qwen2.5:7b model.")

# HuggingFace Token and API key
hf_token = os.getenv('HF_TOKEN')
# api_key = os.getenv("GROQ_API_KEY")

# Verify token and API key
if hf_token:
    # Initialize the LLM
    # llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=api_key)
    # llm = ChatOllama(model="qwen2.5:7b")
    # Initialize the HuggingFace LLM
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # repo_id = "ibm-granite/granite-3.0-8b-instruct"
    # repo_id = "ibm-granite/granite-3.0-2b-instruct"
    # repo_id = "mosaicml/mpt-7b-instruct"

    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=250, temperature=0.7, token=hf_token)

    # HuggingFace Embedding Model for Document Retrieval
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Chat session ID
    session_id = st.text_input("Session ID", value="default_session")

    # Manage chat history in session state
    if "store" not in st.session_state:
        st.session_state.store = {}

    # URLs of real estate websites to scrape data from
    urls = [
        "https://housing.com/in/buy/real-estate-pune",
        "https://www.magicbricks.com/property-for-sale-rent-in-Pune/residential-real-estate-Pune",
        "https://www.squareyards.com/pune-real-estate",
        "https://www.makaan.com/pune-residential-property/buy-property-in-pune-city",
        "https://www.indiaproperty.com/",
        "https://www.commonfloor.com/",
        "https://www.realestateindia.com/property-for-sale-rent-in-india.htm",
     
    ]

    # Scrape data from websites
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

    # # HuggingFace Embedding Model for Document Retrieval
    # embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # # text text_splitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    # splits = text_splitter.split_documents(data)

    # Create FAISS VectorStore from the document splits
    vectorstore = FAISS.from_texts(data, embedding=embedding)

    # Contextualize query with chat history
    # Contextualize query with chat history
    contextualize_q_system_prompt = """
    You are a knowledgeable real estate assistant. Given the chat history and the latest user query, extract the key information from the user input to formulate a new query.

    - Ensure the new query is standalone and contains necessary details such as BHK configuration, location or property.location or situated_at or located_at, price and property.name or Property_Name or.
    - If there is missing context, use previous conversation history to fill in any gaps.
    - If no relevant history or data is found, return "I don't know."

    Your goal is to ensure users get meaningful property search results based on their input and history.
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human', "{input}")
        ]
    )

    parser = StrOutputParser()

    # History-aware retriever
    def create_history_aware_retriever(llm, retriever, system_prompt,parser):
        # Code to create history-aware retriever can go here (custom based on LangChain)
        return retriever

    # History-aware retriever
    history_aware_retrieve = create_history_aware_retriever(llm, vectorstore.as_retriever(), contextualize_q_system_prompt, parser)

    # System prompt for real estate query
    system_prompt = """
    You are an expert real estate property search assistant with 30 years of experience for question-answering task. Your task is to help users find properties as per their requests, and keep the answer concise..
    Get the Properties from the data and show them. get the similar data too. 
    - Provide at least 5 relevant property options if available.
    - If exact matches are unavailable, suggest nearby alternatives within 10 KM of the location provided by the user.
    - If no properties are found, provide guidance on how to search further or check more sources.
    - Answer in the following format for each property,
    - Providing me with the details of your preferences for a property:{context}
        - **Builder/Group Name**: if we have property.name or builder name or group name from the website of the property provide else don't show the tag. property.name, builder_name, by_group_name or else don't show
        - **Location**: location provided by user or closest match found
        - **BHK Configuration**: bhk provided by user or closest match found
        - **Property_Type**: property type provided by user or closest match found
        - **Price**: price provided by user or closest match found
        - **Description**: Provide brief details of property features, nearby facilities, and additional details.

    Always prioritize the user’s preferences for location, BHK configuration, and price, but offer alternatives if needed.
    """

    question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human', "{input}")
        ]
    )

    # Question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, question_prompt)  # Custom implementation here

    # Retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retrieve, question_answer_chain)
    
    if not rag_chain:
            st.error("Error initializing chain, check your configuration.")
            st.stop()

    store = {}
    # Get session history
    # def get_session_history(session: str) -> ChatMessageHistory:
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()  # Initialize if not existing
        return st.session_state.store[session]

        
    # History-enabled conversational chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",         # Key for user input
        history_messages_key="chat_history",# Key for chat history
        output_messages_key="answer"        # Key for the model's response
    )

    st.subheader("**Few :green[Query] Examples**")
    st.write("""
    - Show me the property of 2 BHK in Pune city of Kothrud location, price starting from ₹50 lakhs with a description.
    - Show me 3 BHK flats in Baner, Pune, with a price range of ₹75 lakhs and above, including amenities and features.
    - Find me a 1 BHK flat in Wakad, Pune, under ₹40 lakhs, and provide details about nearby schools and markets.
    - I’m looking for a 4 BHK villa in Kharadi, Pune, with a price above ₹1 crore. Show the builder name and furnishing details.
    - Give me at least 5 options for properties around Baner, Pune, with prices starting from ₹60 lakhs.
    """)
    
    

    # Function to update and append results to a txt file
    def websites_txt_file(results_data, txt_file="real_estate.txt"):
        with open(txt_file, 'a', encoding='utf-8') as file:  # Use 'a' to append to the file instead of 'w'
            file.write("\n\n--- New Search Results ---\n")
            file.write(str(results_data) + "\n")  # Convert dictionary to string
        # st.info(f"Results appended to {txt_file}")

    def response_txt_file(results_data, txt_file="data.txt"):
        with open(txt_file, 'a', encoding='utf-8') as file:  # Use 'a' to append to the file instead of 'w'
            file.write("\n\n--- New Search Results ---\n")
            file.write(str(results_data) + "\n")

    websites_txt_file(data)

    
    # User input
    user_input = st.text_input("Your question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

        # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
    if prompt := st.chat_input(user_input):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})


    # Handle user input and generate response
    if user_input:
        start = time.process_time()
        # st.subheader("🔍 Searching for Your Property...")
        session_history = get_session_history(session_id)  # Maintain history for conversational context

        # Invoke the history-aware chain with session and user input
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        # with st.spinner("Processing..."):
        #     time.sleep(.5)
        #     st.write("Generating response...")
            
        # Display response if available
        if response:
            # st.toast('Yes...')
            # time.sleep(.5)
            # st.toast('We have')
            # time.sleep(.5)
            st.toast("🎉 Found properties for you!")
            placeholder = st.empty()
            placeholder.markdown(":material/search:  Searching for Your Property... ")
            placeholder.progress(0, "Wait for it...")
            time.sleep(1)
            placeholder.progress(50, "Wait for it...")
            time.sleep(1)
            placeholder.progress(100, "Wait for it...")
            placeholder.success(f"Assistant: {response['answer']}")
            response_txt_file(response)  # Append results to a text file for record-keeping

        # Record response time
        end = time.process_time()
        st.info(f"Response time: {end - start:.2f} seconds")
        # st.info(f"Response time: {((end - start)/60)*100:.2f} min")
    
    else:
        st.error("No input provide")

       
# Stop execution if tokens are missing
else:
    st.warning("Please enter the API key")
    st.error("HuggingFace token not found! Please set HF_TOKEN in your .env file.")
    st.stop()




