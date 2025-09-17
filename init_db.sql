-- init_db.sql
CREATE TABLE IF NOT EXISTS facts (
  id SERIAL PRIMARY KEY,
  fact_key VARCHAR(255),
  content JSONB NOT NULL,
  created_by VARCHAR(255),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  version INTEGER DEFAULT 1,
  tombstone BOOLEAN DEFAULT false
);
CREATE INDEX IF NOT EXISTS ix_facts_key_tombstone ON facts (fact_key, tombstone);

CREATE TABLE IF NOT EXISTS audit_log (
  id SERIAL PRIMARY KEY,
  fact_id INTEGER,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
  actor VARCHAR(255),
  channel VARCHAR(50),
  op_type VARCHAR(50) NOT NULL,
  before JSONB,
  after JSONB
);
CREATE INDEX IF NOT EXISTS ix_audit_fact ON audit_log (fact_id);
