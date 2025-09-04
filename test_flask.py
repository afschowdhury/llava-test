#!/usr/bin/env python3
"""
Simple Flask test to check if Flask is working
"""

try:
    from flask import Flask
    print("✓ Flask is installed and working")
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return '<h1>Flask is working!</h1><p>You can now run the video analyzer.</p>'
    
    print("Starting test Flask server on http://localhost:5001")
    print("Visit http://localhost:5001 to test")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
    
except ImportError as e:
    print(f"✗ Flask not installed: {e}")
    print("Please install Flask:")
    print("  pip install flask")
except Exception as e:
    print(f"✗ Error: {e}")
