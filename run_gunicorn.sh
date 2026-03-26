#!/bin/bash
gunicorn --bind 0.0.0.0:${PORT:-8000} "app:create_app()" --workers 1 --log-level debug
