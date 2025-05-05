

from app import app    # <- imports the Flask app object from app.py

# No need for an if __name__ == "__main__" guard here—
# Vercel will detect `app` and serve it as a serverless function.

