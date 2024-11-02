import streamlit as st

"""
  " res.py is a LLM response code, 
  \nrestate.py is a chat_history_response,  
  \nproperty.py is a chatbot_past_history, 
  \nemi.py is EMI"
"""


st.title("About Our Real Estate App")
st.write("""
### About This Application
Our real estate application provides users with an easy-to-use platform to search for properties based on their specific preferences. We combine state-of-the-art language models with real estate data to offer highly accurate and personalized property recommendations.

### Features
- **Intelligent Property Search**: Get results tailored to your preferences.
- **Historical Chat Context**: The app remembers previous chats to provide more relevant responses over time.
- **Location-Based Recommendations**: Offers alternative properties if an exact match isn't available.

### Technology Stack
This application is built with:
- **Streamlit** for a fast, interactive UI.
- **LangChain** for efficient conversational responses.
- **BeautifulSoup** and **Requests** for web scraping.
- **FAISS** for quick property data retrieval.

We hope this app enhances your property search experience!
""")


# st.image("path_to_image/about_us_image.jpg", use_column_width=True)
