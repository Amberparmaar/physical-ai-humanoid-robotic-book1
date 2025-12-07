from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    role = Column(String(50), default='student')
    preferences = Column(Text)  # JSON string
    last_login = Column(DateTime)

class LearningProgress(Base):
    __tablename__ = "learning_progress"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    module_id = Column(String(100), nullable=False)
    content_completed = Column(Integer, default=0)  # percentage
    score = Column(DECIMAL(5, 2))
    time_spent = Column(Integer, default=0)  # in seconds
    last_accessed = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="progress")

User.progress = relationship("LearningProgress", order_by=LearningProgress.created_at, back_populates="user")

class ContentTranslation(Base):
    __tablename__ = "content_translations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    original_content_id = Column(String(255), nullable=False)
    language = Column(String(10), nullable=False)
    translated_content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)