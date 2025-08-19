import streamlit as st
import requests

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

st.title("Harshit's RAG PDF Chatbot")

# Input box for the query
query = st.text_input("Enter your question:")

# Strategy selection (hardcoded or later can fetch dynamically)
strategy = st.selectbox("Select chunking strategy:", [
    "Recursive", "Fixed", "Sliding", "Sentence", "Paragraph",
    "Keyword", "Table", "Topic", "ContentAware", "Semantic", "EmbeddingChunking"
])

# Submit button
if st.button("Ask") or st.session_state.get("submit_on_enter"):
    if query.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Getting answer from RAG..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"query": query, "strategy": strategy}
                ).json()

                st.subheader("Answer:")
                st.write(response.get("answer", "No answer returned."))

                st.subheader("Info:")
                st.write(f"Chunks used: {response.get('chunks_used', 0)} / {response.get('total_chunks', 0)}")
                st.write(f"Sources: {', '.join(response.get('sources', []))}")

            except Exception as e:
                st.error(f"Error fetching answer: {e}")
