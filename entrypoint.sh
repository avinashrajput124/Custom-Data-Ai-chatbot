#!/bin/sh
# Activate the virtual environment
. /app/venv/bin/activate

# Run the ingest script
python /app/ingest.py

# Start the Flask application
exec python /app/app.py
