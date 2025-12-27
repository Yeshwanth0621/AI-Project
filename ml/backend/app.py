from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from groq_service import stream_chat
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')

    if not message:
        return Response("Message is required", status=400)

    def generate():
        for chunk in stream_chat(message):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/', methods=['GET'])
def health_check():
    return "AI Backend is running. Status: Online."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
