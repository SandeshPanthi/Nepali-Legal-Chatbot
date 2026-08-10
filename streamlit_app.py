
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

st.title("Legal RAG Assistant")
st.write("Ask a question about the legal documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input stays at the bottom
query = st.chat_input("Enter your legal question:")

if query:
    # Display user's query
    with st.chat_message("user"):
        st.write(query)

    # Save user's query
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    try:
        response = requests.post(
            API_URL,
            json={"query": query}
        )

        if response.status_code == 200:
            result = response.json()

            answer = result["answer"]

            # Display answer
            with st.chat_message("assistant"):
                st.write(answer)

                st.subheader("Sources")

                for source in result["sources"]:
                    st.write(
                        f"**{source['source']}** | "
                    )

       

            # Save answer
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:
            st.error(
                f"API error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend."
        )

