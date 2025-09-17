# memory_api.py
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from memory import write_fact, read_fact_by_id, search_facts_by_key, forget_fact, get_audit_for_fact

router = APIRouter()
logger = logging.getLogger("ai-sales-agent.memory-api")

class WriteFactRequest(BaseModel):
    content: dict
    created_by: Optional[str] = None
    fact_key: Optional[str] = None

@router.post("/memory/write")
async def api_write_fact(req: WriteFactRequest):
    f = write_fact(req.content, created_by=req.created_by, fact_key=req.fact_key)
    return {"ok": True, "fact": f}

@router.get("/memory/{fid}")
async def api_read_fact(fid: int):
    f = read_fact_by_id(fid)
    if not f:
        raise HTTPException(status_code=404, detail="fact not found")
    return {"ok": True, "fact": f}

@router.get("/memory/search")
async def api_search(key: str):
    rows = search_facts_by_key(key)
    return {"ok": True, "results": rows}

@router.post("/memory/forget/{fid}")
async def api_forget(fid: int):
    ok = forget_fact(fid)
    return {"ok": ok}

@router.get("/memory/audit/{fid}")
async def api_audit(fid: int):
    rows = get_audit_for_fact(fid)
    return {"ok": True, "audit": rows}
