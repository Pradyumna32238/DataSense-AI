import os
from flask import Flask
from api.routes.main import main_bp
from config import setup_logging

def create_app():
    """Application factory to create and configure the Flask app."""
    # Set up logging
    setup_logging()

    app = Flask(__name__, static_folder='static', static_url_path='/static', template_folder='templates')
    
    # Register the main blueprint
    app.register_blueprint(main_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    # This allows running the app directly with 'python app.py' for local development.
    # For production, Gunicorn will be used.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
