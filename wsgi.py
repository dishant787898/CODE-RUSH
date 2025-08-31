import os
import sys

# Try to import the full app, fall back to lite version if there's an error
try:
    from app import app
    print("Running full application with ML models")
except Exception as e:
    print(f"Error loading full app: {e}")
    print("Falling back to lite version...")
    try:
        from app_lite import app
        print("Running lite application with mock predictions")
    except Exception as e:
        print(f"Error loading lite app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)