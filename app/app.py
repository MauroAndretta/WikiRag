import streamlit as st
import sys
from PIL import Image

sys.path.insert(1, "../")

# Import the WikiRag class from the wiki_rag directory
from wiki_rag import WikiRag

# Set page configuration
st.set_page_config(
    page_title="WikiRag Q&A System",
    page_icon="📚",
    layout="centered",
)

# Load an image for the header (optional)
# header_image = Image.open("path_to_your_image.png")
# st.image(header_image, use_column_width=True)

# Initialize the WikiRag class
wiki_rag = WikiRag(
    qdrant_url="http://localhost:6333",  # Adjust as necessary
    qdrant_collection_name="olympics",   # Adjust as necessary
)

# Streamlit application title with custom markdown
st.markdown("<h1 style='text-align: center; color: #F0FFFF;'>WikiRag Q&A System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #777;'>Chiedi qualsiasi cosa sui Giochi Olimpici!</p>", unsafe_allow_html=True)

# User input for the query with placeholder text
user_query = st.text_input("Inserisci la tua domanda:", placeholder="Es. Quale città ha ospitato i primi Giochi Olimpici moderni?")

# Button to submit the query with a custom button style
if st.button("🔍 Chiedi"):
    if user_query:
        with st.spinner("Sto cercando la risposta..."):
            # Get the response from the WikiRag system
            response = wiki_rag.invoke(user_query)

        # Display the response with markdown styling
        st.markdown("### Risposta:")
        st.write(response)
        
    else:
        st.error("Per favore, inserisci una domanda.")

# Display a history of questions and answers
if "history" not in st.session_state:
    st.session_state["history"] = []

if user_query:
    st.session_state["history"].append((user_query, response))

if st.session_state["history"]:
    st.markdown("## Storico delle domande")
    for i, (q, a) in enumerate(st.session_state["history"]):
        with st.expander(f"Domanda #{i+1}: {q}"):
            st.write(f"**Risposta:** {a}")

# Add a footer with a custom message
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Powered by WikiRag | 2024 © Mauro Andretta</p>", unsafe_allow_html=True)
