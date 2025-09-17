# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True, index=True)
    fact_key = Column(String(255), nullable=True, index=True)  # optional friendly key
    content = Column(JSON, nullable=False)  # canonical content JSON
    created_by = Column(String(255), nullable=True)  # actor (e.g., "+919.../voice" or "text")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    version = Column(Integer, default=1)
    tombstone = Column(Boolean, default=False, index=True)

    # index on tombstone + key for fast filtered fetch
    __table_args__ = (Index("ix_facts_key_tombstone", "fact_key", "tombstone"),)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    fact_id = Column(Integer, nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    actor = Column(String(255), nullable=True)  # who (voice/text/system)
    channel = Column(String(50), nullable=True)  # e.g., "voice", "text"
    op_type = Column(String(50), nullable=False)  # create/update/delete/forget
    before = Column(JSON, nullable=True)
    after = Column(JSON, nullable=True)
