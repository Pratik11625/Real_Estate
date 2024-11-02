import streamlit as st
# st.title("Hello")


#page -- setup
about = st.Page(
    page="pages/about.py",
    title="About",
    icon=":material/home:",
    default=True,
)

real_estate = st.Page(
    page="pages/restate.py",
    title="Chat_history_response",
    icon=":material/home:",
    # default=True,
)

real_estate1 = st.Page(
    page="pages/res.py",
    title="LLM_response",
    icon=":material/home:",
    # default=True,
)

real_estate2 = st.Page(
    page="pages/property.py",
    title="chatbot_past_history",
    icon=":material/money:",
    # default=True,
)

emi = st.Page(
    page="pages/emi.py",
    title="EMI",
    icon=":material/money:",
    # default=True,
)

# navigation
# pg = st.navigation(pages=[real_estate, emi])

pg = st.navigation(
    {
        "About_application":[about],
        "Real_estate_property": [real_estate],
        "Real_estate_property1": [real_estate1],
        "Real_estate_property2": [real_estate2],
        "EMI_calculator": [emi],
    }
)

st.logo("download.png",  )
st.sidebar.text("made by kitarp")

pg.run()