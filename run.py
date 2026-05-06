"""Run the AI Roguelite web demo."""
from server.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)
