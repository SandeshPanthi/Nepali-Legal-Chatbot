import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"


st.title("Legal RAG Assistant")
st.write("Ask a question about the legal documents.")


query = st.text_input("Enter your legal question:")


if st.button("Ask"):
    if query:
        try:
            response = requests.post(
                API_URL,
                json={"query": query}
            )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")

                for source in result["sources"]:
                    st.write(
                        f"**{source['source']}** | "
                        f"Page: {source['page']} | "
                        f"Score: {source['score']:.3f}"
                    )

                st.subheader("Confidence")
                st.write(f"{result['confidence']:.3f}")

            else:
                st.error(
                    f"API error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend."
            )