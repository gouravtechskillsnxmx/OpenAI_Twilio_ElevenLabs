# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# Expect DATABASE_URL like: postgresql://user:pass@host:5432/dbname
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set (e.g. postgresql://user:pass@host:5432/db)")

# Use SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
