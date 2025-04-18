from flask import Flask
from app.api.routes import api, initialize_resources
from app.config import settings
import os

def create_app():
    """Create and configure the Flask application."""
    # Create Flask app with the correct template folder
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Register blueprints
    app.register_blueprint(api)
    
    return app

if __name__ == '__main__':
    # Create the application
    app = create_app()
    
    # Initialize resources
    if initialize_resources():
        print("Resources initialized successfully!")
        app.run(
            host=settings.FLASK_HOST,
            port=settings.FLASK_PORT,
            debug=settings.FLASK_DEBUG
        )
    else:
        print("Failed to initialize resources. Make sure all required files exist.")