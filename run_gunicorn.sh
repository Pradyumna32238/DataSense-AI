#!/bin/sh
echo "Starting server with Gunicorn..."

gunicorn --workers 2 \
         --bind 0.0.0.0:$PORT \
         --log-level debug \
         app:app
