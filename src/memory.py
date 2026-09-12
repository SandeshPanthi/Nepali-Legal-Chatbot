"""
Conversation memory backed by SQLite.
"""
import json
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

from config import BASE_DIR

# Store DB in data/ folder
DB_PATH = BASE_DIR / "data" / "conversations.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation",
                            cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey(
        "conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class MemoryManager:
    """Simple conversation memory manager."""

    @staticmethod
    def get_or_create_conversation(db: Session, session_id: str) -> Conversation:
        conv = db.query(Conversation).filter(
            Conversation.session_id == session_id).first()
        if not conv:
            # <-- None, not "New Conversation"
            conv = Conversation(session_id=session_id, title=None)
            db.add(conv)
            db.commit()
            db.refresh(conv)
        return conv

    @staticmethod
    def get_history(db: Session, session_id: str, limit: int = 6) -> List[dict]:
        """Get last N message pairs as dicts for prompt injection."""
        conv = db.query(Conversation).filter(
            Conversation.session_id == session_id).first()
        if not conv:
            return []

        messages = db.query(Message).filter(Message.conversation_id == conv.id).order_by(
            Message.created_at.desc()).limit(limit * 2).all()
        messages.reverse()

        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

    @staticmethod
    def add_message(db: Session, session_id: str, role: str, content: str, sources: Optional[list] = None):
        conv = MemoryManager.get_or_create_conversation(db, session_id)
        msg = Message(
            conversation_id=conv.id,
            role=role,
            content=content,
            sources=json.dumps(sources) if sources else None
        )
        db.add(msg)

        # Auto-title from first real user message
        if role == "user" and (not conv.title or conv.title == "New Conversation"):
            # Clean up for title
            title = content.strip()
            # Remove question marks and extra spaces for cleaner title
            title = title.replace('?', '').replace('!', '')
            if len(title) > 60:
                title = title[:60].rstrip() + "..."
            conv.title = title if title else "New Conversation"

        db.commit()
        return msg

    @staticmethod
    def get_conversations(db: Session, limit: int = 20) -> List[Conversation]:
        return db.query(Conversation).order_by(Conversation.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_conversation_messages(db: Session, session_id: str) -> List[Message]:
        conv = db.query(Conversation).filter(
            Conversation.session_id == session_id).first()
        if not conv:
            return []
        return db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.created_at.asc()).all()

    @staticmethod
    def delete_conversation(db: Session, session_id: str):
        conv = db.query(Conversation).filter(
            Conversation.session_id == session_id).first()
        if conv:
            db.delete(conv)
            db.commit()
            return True
        return False


def format_history_for_prompt(history: List[dict]) -> str:
    """Format conversation history for the LLM prompt."""
    if not history:
        return ""

    lines = []
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")

    return "\n\n".join(lines)