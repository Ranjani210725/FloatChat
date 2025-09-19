# backend.py
from flask import Flask, request, jsonify
from chat_engine import chat_with_cohere  # Your chatbot logic
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response_text, raw_data = chat_with_cohere(user_input)
    return jsonify({"reply": response_text})

if __name__ == '__main__':
    app.run(port=5000)
