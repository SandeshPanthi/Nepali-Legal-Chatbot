import uuid
from fastapi import APIRouter, Request, Depends, Cookie, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from jinja2 import Environment, FileSystemLoader
from markupsafe import escape
import markdown

from api.routes.dependencies import get_db
from src.memory import MemoryManager
from src.rag_pipeline import rag_llm_with_history
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.rag_pipeline import get_llm

router = APIRouter()

# Raw Jinja2 with cache disabled
env = Environment(loader=FileSystemLoader("templates"), cache_size=0)

# Add markdown filter


def md_filter(text):
    """
    Convert LLM markdown output to HTML.

    Two things were silently eating content before:
    1. Raw '<' / '>' in the text (e.g. legal citations like "<Section 5>",
       or comparisons like "income < 5000") were being interpreted as
       unrecognized HTML tags and dropped by the browser. Escaping first
       fixes this.
    2. Nested/sub-bullet lists using 2-space indentation (common in LLM
       output) don't parse under standard python-markdown, which requires
       4-space indents — sane_lists + a normalization pass fixes this.
    """
    if not text:
        return ""

    # 1. Escape raw HTML-like characters so they render as text, not tags
    text = str(escape(text))

    # 2. Normalize common 2-space nested list indentation to 4-space
    #    so python-markdown's list parser picks it up correctly.
    text = _normalize_list_indentation(text)

    return markdown.markdown(
        text,
        extensions=['extra', 'nl2br', 'sane_lists']
    )


def _normalize_list_indentation(text: str) -> str:
    """
    Doubles leading whitespace on list-item lines so 2-space nested
    bullets/numbered items become 4-space, which python-markdown's
    parser requires to recognize nesting.
    """
    import re

    lines = text.split("\n")
    fixed = []
    list_item_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')

    for line in lines:
        match = list_item_pattern.match(line)
        if match:
            leading_space = match.group(1)
            if leading_space:
                # Double the indentation (2-space -> 4-space, etc.)
                new_indent = leading_space * 2
                line = new_indent + line[len(leading_space):]
        fixed.append(line)

    return "\n".join(fixed)


env.filters['md'] = md_filter

# Initialize RAG components
embedding_manager = EmbeddingManager()
vector_store = VectorStore()
retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)
llm = get_llm()


def render(name: str, **context) -> HTMLResponse:
    template = env.get_template(name)
    return HTMLResponse(content=template.render(**context))


@router.get("/")
def chat_page(request: Request, session_id: str = Cookie(None), db: Session = Depends(get_db)):
    if not session_id:
        session_id = str(uuid.uuid4())

    messages = MemoryManager.get_conversation_messages(db, session_id)
    conversations = MemoryManager.get_conversations(db, limit=15)

    response = render(
        "chat.html",
        request=request,
        messages=messages,
        conversations=conversations,
        current_session=session_id
    )

    if not request.cookies.get("session_id"):
        response.set_cookie(
            key="session_id", value=session_id, max_age=2592000)

    return response


@router.post("/chat")
def chat(
    request: Request,
    query: str = Form(...),
    session_id: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not session_id:
        session_id = str(uuid.uuid4())

    MemoryManager.add_message(db, session_id, "user", query)
    history = MemoryManager.get_history(db, session_id, limit=3)

    result = rag_llm_with_history(
        query=query,
        retriever=retriever,
        llm=llm,
        history=history
    )

    MemoryManager.add_message(
        db, session_id, "assistant",
        result["answer"],
        sources=result.get("sources")
    )

    return render(
        "partials/message.html",
        request=request,
        query=query,
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )


@router.post("/new-chat")
def new_chat(request: Request, db: Session = Depends(get_db)):
    new_session = str(uuid.uuid4())
    conversations = MemoryManager.get_conversations(db, limit=15)

    response = render(
        "chat.html",
        request=request,
        messages=[],
        conversations=conversations,
        current_session=new_session
    )
    response.set_cookie(key="session_id", value=new_session, max_age=2592000)
    return response


@router.get("/history/{session_id}")
def load_conversation(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db)
):
    messages = MemoryManager.get_conversation_messages(db, session_id)
    conversations = MemoryManager.get_conversations(db, limit=15)

    response = render(
        "chat.html",
        request=request,
        messages=messages,
        conversations=conversations,
        current_session=session_id
    )
    response.set_cookie(key="session_id", value=session_id, max_age=2592000)
    return response


@router.delete("/history/{session_id}")
def delete_conversation(
    session_id: str,
    db: Session = Depends(get_db)
):
    MemoryManager.delete_conversation(db, session_id)
    return {"status": "deleted"}