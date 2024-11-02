import streamlit as st

# st.set_page_config(
#     page_title="About Us",
#      page_icon="ℹ️",
#      initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# This is a header. This is an *extremely* cool app!"
#     })

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
