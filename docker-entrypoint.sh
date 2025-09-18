#!/usr/bin/env bash
set -euo pipefail

# Helpful logging
echo "[entrypoint] starting entrypoint. INIT_DB='${INIT_DB:-}'"

# If INIT_DB=true, run SQL init using psql and DATABASE_URL from env
if [ "${INIT_DB:-false}" = "true" ]; then
  if [ -z "${DATABASE_URL:-}" ]; then
    echo "[entrypoint] ERROR: DATABASE_URL is not set. Cannot run init_db.sql" >&2
  else
    echo "[entrypoint] INIT_DB=true: running /app/init_db.sql against DATABASE_URL"
    # Ensure psql understands SSL requirement — many managed DBs require sslmode=require
    # If the URL already contains "?" query params, append with &; otherwise append ?sslmode=require
    if echo "$DATABASE_URL" | grep -q "sslmode="; then
      PSQL_URL="$DATABASE_URL"
    else
      # append sslmode=require safely (if DATABASE_URL contains query part)
      if echo "$DATABASE_URL" | grep -q "?" ; then
        PSQL_URL="${DATABASE_URL}&sslmode=require"
      else
        PSQL_URL="${DATABASE_URL}?sslmode=require"
      fi
    fi

    echo "[entrypoint] using psql connection string with ssl; running psql..."
    # Run psql to execute init script. psql accepts full connection URL.
    psql "$PSQL_URL" -f /app/init_db.sql \
      && echo "[entrypoint] init_db.sql executed successfully" \
      || { echo "[entrypoint] psql reported an error"; exit 1; }
  fi
fi

# If first argument(s) passed to container, run them (so you can override CMD when needed)
if [ "$#" -gt 0 ]; then
  echo "[entrypoint] exec: $@"
  exec "$@"
else
  # If no args, fall back to default CMD from Dockerfile
  exec "$@"
fi
