"""
Query router: decides whether a question needs Nepali legal document retrieval.
"""


def classify_query(query: str, llm) -> str:
    """
    Uses the LLM to classify the user intent.
    Returns: 'LEGAL' or 'GENERAL'
    """
    router_prompt = f"""You are a query classifier for a Nepali Legal Assistant chatbot.

Your job is to decide whether the user's message requires searching Nepali legal documents (Constitution, Civil Code, Penal Code) to answer correctly.

Classify the query as:
- LEGAL: if it's about Nepali law, legal procedures, rights, crimes, constitution, civil code, penal code, court procedures, contracts, property, marriage, divorce, inheritance, citizenship, or any legal matter in Nepal.
- GENERAL: if it's casual conversation, greetings, personal questions, math, science, history, jokes, general knowledge, or anything NOT specifically about Nepali law.

Reply with exactly one word: LEGAL or GENERAL.

User query: {query}

Classification:"""

    response = llm.invoke(router_prompt)
    classification = response.content.strip().upper()

    if "LEGAL" in classification:
        return "LEGAL"
    return "GENERAL"