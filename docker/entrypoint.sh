#!/bin/sh
set -e

echo "Checking ChromaDB..."

if [ ! -f /app/chroma_db/chroma.sqlite3 ]; then
    echo "Building vector index..."
    python -m src.ingest.build_index
else
    echo "Existing ChromaDB found."
fi

echo "Starting backend..."
exec "$@"