import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from streamlit_chat import message


# Load environment variables (like HuggingFace token and API key)
load_dotenv()

# Initialize Streamlit UI
st.title("🏠 Real Estate Property Search with Chat History")
# st.markdown("Using the qwen2.5:7b model.")

# Load HuggingFace token
hf_token = os.getenv('HF_TOKEN')
groq_api_key = os.getenc("GROQ_API_KEY")

if not hf_token and groq_api_key:
    st.error("HuggingFace token not found! Please set HF_TOKEN in your .env file.")
    st.stop()

# Set up model and embedding
# llm = ChatOllama(model="qwen2.5:7b")

# Initialize the HuggingFace LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
repo_id = "Qwen/Qwen2.5-72B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=150,
    temperature=0.7,
    token=hf_token
)

# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model="llama3-groq-70b-8192-tool-use-preview",
# )

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define URLs for scraping property data
urls = [
    "https://housing.com/in/buy/real-estate-pune",
    "https://www.magicbricks.com/property-for-sale-rent-in-Pune/residential-real-estate-Pune",
    "https://www.squareyards.com/pune-real-estate",
]

# Scrape property data
data = []
for url in urls:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            data.append(soup.get_text())
        else:
            st.warning(f"Failed to retrieve data from {url}")
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")

if not data:
    st.error("No data loaded from any of the URLs.")
    st.stop()

# Create a vector store for property data
vectorstore = FAISS.from_texts(data, embedding=embedding)

# Define system prompts for question reformulation
contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a knowledgeable real estate assistant. Given the chat history and the latest user query, extract the key information from the user input to formulate a new query.

        - Ensure the new query is standalone and contains necessary details such as BHK configuration, location or property.location or situated_at or located_at, price and property.name or Property_Name or.
        - If there is missing context, use previous conversation history to fill in any gaps.
        - If no relevant history or data is found, return "I don't know."

        Your goal is to ensure users get meaningful property search results based on their input and history.
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.
        """),
        MessagesPlaceholder("chat_history"),
        ('human', "{input}")
    ]
)

# Define the main question prompt for property searching
question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
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
    """),
        MessagesPlaceholder("chat_history"),
        ('human', "{input}")
    ]
)

# Define the conversational chain with history
def create_history_aware_retriever(llm, retriever, system_prompt):
    # Placeholder for any custom history-aware retrieval logic
    return retriever

history_aware_retrieve = create_history_aware_retriever(llm, vectorstore.as_retriever(), contextualize_q_system_prompt)
question_answer_chain = create_stuff_documents_chain(llm, question_prompt)
conversational_rag_chain = create_retrieval_chain(history_aware_retrieve, question_answer_chain)

# Function to retrieve or initialize chat history
def get_session_history(session: str) -> ChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# History-enabled conversational chain
conversational_rag_chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)



# import streamlit as st
def display_conversation(history):
        for i in range(len(history["generated"])):
            st.write(f"User: {history['past'][i]}")
            st.write(f"Assistant: {history['generated'][i]}")

# Initialize session and chat history
session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
        st.session_state.store = {}
if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hi! How can I assist you with property searches today?"]
if "past" not in st.session_state:
        st.session_state["past"] = ["Hello!"]

    # Initialize ChatMessageHistory for the session if not done
if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    # Loop to handle user input
while True:
        user_input = st.text_input("Ask your question (type 'quit' to exit)")
        if user_input.lower() == 'quit':
            st.write("Ending session.")
            break
        if user_input:
            # Add `user input to chat history using add_user_message()
             # Invoke the history-aware chain with session and user input
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.success(f"Assistant: {response['answer']}")
            session_history = st.session_state.store[session_id]
            session_history.add_user_message(user_input)
            
            # Retrieve the chat history as a list of messages
            chat_history = [
                {"role": "user", "content": msg.content} if msg.type == "user" else {"role": "assistant", "content": msg.content}
                for msg in session_history.messages
            ]
            
            # Invoke history-aware chain
            response = conversational_rag_chain.invoke(
                {"input": user_input, "chat_history": chat_history},
                config={"session_id": session_id}
            )
            
            # Append response to chat history
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response['answer'])
            session_history.add_ai_message(response['answer'])

            # Display conversation
            display_conversation({"past": st.session_state["past"], "generated": st.session_state["generated"]})



# # Display conversation function
# def display_conversation():
#     for i in range(len(st.session_state.past)):
#         st.chat_message("user").markdown(st.session_state.past[i])
#         st.chat_message("assistant").markdown(st.session_state.generated[i])

# # Initialize session variables if not present
# session_id = st.text_input("Session ID", value="default_session")
# if "store" not in st.session_state:
#     st.session_state.store = {}
# if "generated" not in st.session_state:
#     st.session_state["generated"] = ["Hi! How can I assist you with property searches today?"]
# if "past" not in st.session_state:
#     st.session_state["past"] = ["Hello!"]

# # Main chat loop
# while True:
#     user_input = st.text_input("Ask your question (type 'quit' to exit)")

#     # End session if user types 'quit'
#     if user_input.lower() == 'quit':
#         st.write("Ending session.")
#         break

#     if user_input:
#         # Add user message to chat history
#         session_history = get_session_history(session_id)
#         session_history.add_user_message(user_input)
#         st.session_state.past.append(user_input)
        
#         # Invoke conversational chain
#         response = conversational_rag_chain.invoke({"input": user_input}, config={"session_id": session_id})
#         assistant_response = response['answer']
        
#         # Add assistant's response to chat history
#         st.session_state.generated.append(assistant_response)
#         session_history.add_ai_message(assistant_response)
        
#         # Display the updated conversation
#         display_conversation()

# -------------------------------------------------------------------------------------------------------------
# # import os
# # import streamlit as st
# # import pandas as pd
# # import requests
# # from bs4 import BeautifulSoup
# # import time
# # from dotenv import load_dotenv

# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.vectorstores import FAISS
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_ollama import ChatOllama
# # from langchain_community.chat_message_histories import ChatMessageHistory

# # Load environment variables (like HuggingFace token and API key)
# load_dotenv()

# # Initialize Streamlit UI
# st.title("🏠 Real Estate Property Search with Chat History")
# st.markdown("Using the qwen2.5:7b model.")

# # Set up HuggingFace token and API key
# hf_token = os.getenv('HF_TOKEN')
# api_key = os.getenv("GROQ_API_KEY")

# # Verify token and API key
# # if hf_token and api_key == True:
# if hf_token:
#     llm = ChatOllama(model="qwen2.5:7b")
#     embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Initialize session and chat history
#     session_id = st.text_input("Session ID", value="default_session")
#     if "store" not in st.session_state:
#         st.session_state.store = {}
#     if "generated" not in st.session_state:
#         st.session_state["generated"] = ["Hi! How can I assist you with property searches today?"]
#     if "past" not in st.session_state:
#         st.session_state["past"] = ["Hello!"]

#     # Define URLs for scraping property data
#     urls = [
#         "https://housing.com/in/buy/real-estate-pune",
#         "https://www.magicbricks.com/property-for-sale-rent-in-Pune/residential-real-estate-Pune",
#         "https://www.squareyards.com/pune-real-estate",
#         # Add more URLs if needed
#     ]

#     # Scrape property data
#     data = []
#     for url in urls:
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.content, "html.parser")
#                 data.append(soup.get_text())
#             else:
#                 st.warning(f"Failed to retrieve data from {url}")
#         except Exception as e:
#             st.error(f"Error loading data from {url}: {e}")

#     if not data:
#         st.error("No data loaded from any of the URLs.")
#         st.stop()

#     # Create a vector store for property data
#     vectorstore = FAISS.from_texts(data, embedding=embedding)

#     # Define system prompts for question reformulation
#     contextualize_q_system_prompt = """
#     You are a knowledgeable real estate assistant. Given the chat history and the latest user query, extract the key information from the user input to formulate a new query.

#     - Ensure the new query is standalone and contains necessary details such as BHK configuration, location or property.location or situated_at or located_at, price and property.name or Property_Name or.
#     - If there is missing context, use previous conversation history to fill in any gaps.
#     - If no relevant history or data is found, return "I don't know."

#     Your goal is to ensure users get meaningful property search results based on their input and history.
#     Given a chat history and the latest user question \
#     which might reference context in the chat history, formulate a standalone question \
#     which can be understood without the chat history. Do NOT answer the question, \
#     just reformulate it if needed and otherwise return it as is.
#     """
#     contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ('human', "{input}")
#         ]
#     )

#     parser = StrOutputParser()

#     # Function to create a history-aware retriever
#     def create_history_aware_retriever(llm, retriever, system_prompt):
#         return retriever  # Add your custom logic if needed

#     # Define the property search prompt
#     system_prompt = """
#     You are an expert real estate property search assistant with 30 years of experience for question-answering task. Your task is to help users find properties as per their requests, and keep the answer concise..
#     Get the Properties from the data and show them. get the similar data too. 
#     - Provide at least 5 relevant property options if available.
#     - If exact matches are unavailable, suggest nearby alternatives within 10 KM of the location provided by the user.
#     - If no properties are found, provide guidance on how to search further or check more sources.
#     - Answer in the following format for each property,
#     - Providing me with the details of your preferences for a property:{context}
#         - **Builder/Group Name**: if we have property.name or builder name or group name from the website of the property provide else don't show the tag. property.name, builder_name, by_group_name or else don't show
#         - **Location**: location provided by user or closest match found
#         - **BHK Configuration**: bhk provided by user or closest match found
#         - **Property_Type**: property type provided by user or closest match found
#         - **Price**: price provided by user or closest match found
#         - **Description**: Provide brief details of property features, nearby facilities, and additional details.

#     Always prioritize the user’s preferences for location, BHK configuration, and price, but offer alternatives if needed.
#     """
#     question_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ('human', "{input}")
#         ]
#     )

#     # Define main conversational chain
#     history_aware_retrieve = create_history_aware_retriever(llm, vectorstore.as_retriever(), contextualize_q_system_prompt)
#     question_answer_chain = create_stuff_documents_chain(llm, question_prompt)
#     conversational_rag_chain = create_retrieval_chain(history_aware_retrieve, question_answer_chain)

#     def get_session_history(session: str) -> BaseChatMessageHistory:
#             if session not in st.session_state.store:
#                 st.session_state.store[session] = ChatMessageHistory()  # Initialize if not existing
#             return st.session_state.store[session]

#     # History-enabled conversational chain
#     conversational_rag_chain = RunnableWithMessageHistory(
#         conversational_rag_chain,
#         get_session_history,
#         input_messages_key="input",         # Key for user input
#         history_messages_key="chat_history",# Key for chat history
#         output_messages_key="answer"        # Key for the model's response
#     )

#     def display_conversation(history):
#         for i in range(len(history["generated"])):
#             st.write(f"User: {history['past'][i]}")
#             st.write(f"Assistant: {history['generated'][i]}")

#         # Replace with actual import

#     # Initialize session variables if not present
#     if "store" not in st.session_state:
#         st.session_state.store = {}

#     if "past" not in st.session_state:
#         st.session_state.past = []

#     if "generated" not in st.session_state:
#         st.session_state.generated = []

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Initialize chat history for the session
#     session_id = "default_session"  # Or get this from user input
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()

#     session_history = st.session_state.store[session_id]

#     # Function to display chat history
#     def display_conversation():
#         for i in range(len(st.session_state.past)):
#             st.chat_message("user").markdown(st.session_state.past[i])
#             st.chat_message("assistant").markdown(st.session_state.generated[i])

#     # Main chat loop
#     while True:
#         user_input = st.text_input("Ask your question (type 'quit' to exit)")

#         # End session if user types 'quit'
#         if user_input.lower() == 'quit':
#             st.write("Ending session.")
#             break

#         # Handle user input
#         if user_input:
#             # Add user message to chat history
#             session_history.add_user_message(user_input)
#             st.session_state.past.append(user_input)
            
#             # Invoke conversational chain (replace with actual invocation)
#             response = conversational_rag_chain.invoke({"input": user_input}, config={"session_id": session_id})
#             assistant_response = response['answer']
            
#             # Add assistant's response to chat history
#             st.session_state.generated.append(assistant_response)
#             session_history.add_ai_message(assistant_response)
            
#             # Display the updated conversation
#             display_conversation()


#     # # Loop to handle user input
#     # while True:
#     #     user_input = st.text_input("Ask your question (type 'quit' to exit)")
#     #     if user_input.lower() == 'quit':
#     #         st.write("Ending session.")
#     #         break
#     #     if user_input:
#     #         # Add user input to chat history
#     #         # session_history = ChatMessageHistory() if session_id not in st.session_state.store else st.session_state.store[session_id]
#     #         # session_history.append({"role": "user", "content": user_input})
#     #          # Add user input to chat history using add_user_message()
#     #         session_history = st.session_state.store[session_id]
#     #         session_history.add_user_message(user_input)
            
#     #         # Retrieve the chat history as a list of messages
#     #         chat_history = [
#     #             {"role": "user", "content": msg.content} if msg.type == "user" else {"role": "assistant", "content": msg.content}
#     #             for msg in session_history.messages
#     #         ]

#     #         # Invoke history-aware chain
#     #         response = conversational_rag_chain.invoke({"input": user_input}, config={"session_id": session_id})
            
#     #         # Append response to chat history
#     #         st.session_state["past"].append(user_input)
#     #         st.session_state["generated"].append(response['answer'])
#     #         session_history.add_ai_message(response['answer'])

#     #         display_conversation({"past": st.session_state["past"], "generated": st.session_state["generated"]})
        
#     #     # Initialize ChatMessageHistory for the session if not done
#     #         if session_id not in st.session_state.store:
#     #             st.session_state.store[session_id] = ChatMessageHistory()

#     #             # Initialize chat history if not present
#     #         if "messages" not in st.session_state:
#     #             st.session_state.messages = []

#     #         # Display chat messages from history on app rerun
#     #         for message in st.session_state.messages:
#     #             with st.chat_message(message["role"]):
#     #                 st.markdown(message["content"])

#     #         # Capture user input
#     #         if user_input := st.chat_input("Enter your message:"):
#     #             # Display user message in chat message container
#     #             st.chat_message("user").markdown(user_input)
#     #             # Add user message to chat history
#     #             st.session_state.messages.append({"role": "user", "content": user_input})

#     #             # Generate a response (replace this with your model or retrieval function)
#     #             response = conversational_rag_chain.invoke({"input": user_input}, config={"session_id": session_id}) # Replace this line with actual response generation logic
#     #             # Display the assistant's response
#     #             with st.chat_message("assistant"):
#     #                 st.markdown(response['answer'])
#     #             # Add assistant's response to chat history
#     #             st.session_state.messages.append({"role": "assistant", "content": response})

#     #             # Append response to chat history
#     #             st.session_state["past"].append(user_input)
#     #             st.session_state["generated"].append(response['answer'])
#     #             session_history.add_ai_message(response['answer'])

#     #             display_conversation({"past": st.session_state["past"], "generated": st.session_state["generated"]})

            
# else:
#     st.warning("Please enter the API key")
#     st.error("HuggingFace token not found! Please set HF_TOKEN in your .env file.")
#     st.stop()
