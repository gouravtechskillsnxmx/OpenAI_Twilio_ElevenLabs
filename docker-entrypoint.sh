#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting entrypoint. INIT_DB='${INIT_DB:-}'"

if [ "${INIT_DB:-false}" = "true" ]; then
  if [ -z "${DATABASE_URL:-}" ]; then
    echo "[entrypoint] ERROR: DATABASE_URL is not set. Cannot run init_db.sql" >&2
  else
    echo "[entrypoint] INIT_DB=true: running /app/init_db.sql against DATABASE_URL"

    # Append sslmode=require unless already present
    if echo "$DATABASE_URL" | grep -q "sslmode="; then
      PSQL_URL="$DATABASE_URL"
    else
      if echo "$DATABASE_URL" | grep -q "?" ; then
        PSQL_URL="${DATABASE_URL}&sslmode=require"
      else
        PSQL_URL="${DATABASE_URL}?sslmode=require"
      fi
    fi

    echo "[entrypoint] using psql connection string with ssl; running psql..."
    if ! command -v psql >/dev/null 2>&1; then
      echo "[entrypoint] ERROR: psql not found in PATH" >&2
      exit 1
    fi

    psql "$PSQL_URL" -f /app/init_db.sql \
      && echo "[entrypoint] init_db.sql executed successfully" \
      || { echo "[entrypoint] psql reported an error"; exit 1; }
  fi
fi

# If arguments were provided to the container, exec them;
# otherwise exec the default CMD (uvicorn) provided by Dockerfile.
if [ "$#" -gt 0 ]; then
  echo "[entrypoint] exec: $@"
  exec "$@"
else
  # No args passed; exec the CMD from Dockerfile
  exec "$@"
fi
