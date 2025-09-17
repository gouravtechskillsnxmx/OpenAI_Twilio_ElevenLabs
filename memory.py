# memory.py
import logging
import json
import time
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session
from models import Fact, AuditLog
from db import SessionLocal
import redis
import os

logger = logging.getLogger("ai-sales-agent.memory")

# Redis config
REDIS_URL = os.environ.get("REDIS_URL")  # e.g., redis://:password@host:6379/0
redis_client = None
if REDIS_URL:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)

CACHE_TTL = int(os.environ.get("MEMORY_CACHE_TTL", "300"))  # seconds

def _cache_key_for_fact(fid: int) -> str:
    return f"fact:{fid}"

def write_fact(content: Dict[str, Any], created_by: Optional[str]=None, fact_key: Optional[str]=None) -> Dict[str, Any]:
    """
    Create a new fact and append an audit entry.
    Returns the fact dict.
    """
    db: Session = SessionLocal()
    try:
        fact = Fact(content=content, created_by=created_by, fact_key=fact_key)
        db.add(fact)
        db.commit()
        db.refresh(fact)

        # audit entry
        audit = AuditLog(fact_id=fact.id, actor=created_by, channel="voice" if created_by and created_by.startswith("+") else "text", op_type="create", before=None, after=content)
        db.add(audit)
        db.commit()

        # cache
        if redis_client:
            redis_client.set(_cache_key_for_fact(fact.id), json.dumps({"id":fact.id, "content":content, "fact_key":fact_key}), ex=CACHE_TTL)

        return {"id": fact.id, "content": content, "fact_key": fact_key}
    finally:
        db.close()

def read_fact_by_id(fid: int) -> Optional[Dict[str, Any]]:
    # check cache
    if redis_client:
        key = _cache_key_for_fact(fid)
        raw = redis_client.get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                logger.exception("Failed to decode cached fact %s", fid)

    db: Session = SessionLocal()
    try:
        fact = db.query(Fact).filter(Fact.id == fid, Fact.tombstone == False).first()
        if not fact:
            return None
        result = {"id": fact.id, "content": fact.content, "fact_key": fact.fact_key}
        if redis_client:
            redis_client.set(_cache_key_for_fact(fid), json.dumps(result), ex=CACHE_TTL)
        return result
    finally:
        db.close()

def search_facts_by_key(key: str) -> List[Dict[str, Any]]:
    db: Session = SessionLocal()
    try:
        rows = db.query(Fact).filter(Fact.fact_key == key, Fact.tombstone == False).order_by(Fact.created_at.desc()).all()
        return [{"id": r.id, "content": r.content, "fact_key": r.fact_key} for r in rows]
    finally:
        db.close()

def forget_fact(fid: int, actor: Optional[str]=None) -> bool:
    """
    Tombstone a fact and append audit. Also invalidate cache and enqueue propagation.
    """
    db: Session = SessionLocal()
    try:
        fact = db.query(Fact).filter(Fact.id == fid).first()
        if not fact or fact.tombstone:
            return False
        before = {"id": fact.id, "content": fact.content, "fact_key": fact.fact_key, "tombstone": fact.tombstone}
        fact.tombstone = True
        fact.version = fact.version + 1 if fact.version else 1
        db.add(fact)
        db.commit()

        audit = AuditLog(fact_id=fid, actor=actor, channel="voice" if actor and actor.startswith("+") else "text", op_type="forget", before=before, after={"tombstone": True})
        db.add(audit)
        db.commit()

        # invalidate cache
        if redis_client:
            try:
                redis_client.delete(_cache_key_for_fact(fid))
            except Exception:
                logger.exception("Failed to delete cache for fact %s", fid)

        # enqueue propagation — for now, call a local propagation function (in future: message queue)
        try:
            _propagate_forget(fid)
        except Exception:
            logger.exception("Propagate forget failed for %s", fid)

        return True
    finally:
        db.close()

def get_audit_for_fact(fid: int):
    db: Session = SessionLocal()
    try:
        rows = db.query(AuditLog).filter(AuditLog.fact_id == fid).order_by(AuditLog.timestamp.asc()).all()
        return [{"timestamp": r.timestamp.isoformat(), "actor": r.actor, "channel": r.channel, "op_type": r.op_type, "before": r.before, "after": r.after} for r in rows]
    finally:
        db.close()

# Stub for propagation to vector DB, caches, etc.
def _propagate_forget(fid: int):
    """
    This function should remove vectors from vector DB and notify caches.
    Here it's synchronous and fast for Booster testing. Replace with async job to meet SLAs at scale.
    """
    logger.info("Propagating forget for fact %s (stub).", fid)
    # TODO: implement vector DB deletion / reindexing / cache invalidation across cluster
    time.sleep(0.1)
    return True
