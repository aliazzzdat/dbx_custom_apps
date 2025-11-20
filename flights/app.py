"""
Airport Management System - Main Application Entry Point

This application provides a comprehensive airport management system with:
- Flight tracking and management
- Passenger records
- Airport and airline management
- Aircraft inventory
- Real-time analytics and KPIs
"""

from flask import Flask
from app.database import initialize_database
from app.routes import init_routes
from app.config import FLASK_DEBUG, FLASK_RUN_HOST, FLASK_RUN_PORT

# Create Flask application
app = Flask(__name__)

# Initialize database with sample data (if needed)
print("Initializing database...")
initialize_database()

# Register all routes
print("Initializing routes...")
init_routes(app)

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("✈️  Airport Management System")
    print(f"{'='*60}")
    print(f"Starting server on http://{FLASK_RUN_HOST}:{FLASK_RUN_PORT}")
    print(f"Debug mode: {FLASK_DEBUG}")
    print(f"{'='*60}\n")
    
    app.run(
        host=FLASK_RUN_HOST,
        port=FLASK_RUN_PORT,
        debug=FLASK_DEBUG
    )
