"""
Database models for the Forex Trading AI System
Stores evolution results, training sessions, and chromosome data
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class TrainingSession(Base):
    """Main training session record"""
    __tablename__ = 'training_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)  # Training configuration
    status = Column(String(50), default='running')  # running, completed, failed
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    # Final results
    best_fitness = Column(Float)
    total_generations = Column(Integer)
    final_generation = Column(Integer)
    
    # Relationships
    generations = relationship("Generation", back_populates="session", cascade="all, delete-orphan")
    training_results = relationship("TrainingResult", back_populates="session", cascade="all, delete-orphan")

class Generation(Base):
    """Individual generation data from evolution"""
    __tablename__ = 'generations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id'), nullable=False)
    generation_number = Column(Integer, nullable=False)
    
    # Generation statistics
    best_fitness = Column(Float, nullable=False)
    avg_fitness = Column(Float, nullable=False)
    max_fitness = Column(Float, nullable=False)
    diversity = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="generations")
    chromosomes = relationship("Chromosome", back_populates="generation", cascade="all, delete-orphan")

class Chromosome(Base):
    """Individual chromosome data"""
    __tablename__ = 'chromosomes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    generation_id = Column(UUID(as_uuid=True), ForeignKey('generations.id'), nullable=False)
    chromosome_index = Column(Integer, nullable=False)
    
    fitness = Column(Float, nullable=False)
    is_champion = Column(Boolean, default=False)
    genes_data = Column(JSON, nullable=False)  # Array of gene objects
    
    # Relationships
    generation = relationship("Generation", back_populates="chromosomes")

class TrainingResult(Base):
    """DQN training results and performance metrics"""
    __tablename__ = 'training_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id'), nullable=False)
    
    # Performance metrics
    episode_rewards = Column(JSON)  # Array of episode rewards
    final_performance = Column(JSON)  # Final performance metrics
    agent_stats = Column(JSON)  # DQN agent statistics
    
    # Feature information
    best_features = Column(JSON)  # Array of selected features
    feature_columns = Column(JSON)  # Array of feature column names
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="training_results")

# Database connection and session management
DATABASE_URL = os.environ.get('DATABASE_URL', '')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let caller handle it

def close_db(db):
    """Close database session"""
    db.close()