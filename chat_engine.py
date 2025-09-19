import cohere
import os
from dotenv import load_dotenv
from llm_backend import generate_response

load_dotenv()
co = cohere.Client(os.getenv("CO_API_KEY"))

def chat_with_cohere(user_input):
    normalized = user_input.strip().lower()

    # Handle casual greetings separately
    if normalized in ["hi", "hello", "hey"]:
        return "Hi! I’m your ocean data assistant. Ask me about temperature, salinity, float locations, or BGC sensors 🌊", None

    # Generate relevant ARGO data
    data_response = generate_response(user_input)

    # Focused prompt for short, clear replies
    prompt = f"""
You are an ocean data assistant. Based on the following ARGO float data, respond to the user's question with a short, clear, and natural answer. Avoid markdown, long intros, or formatting. Keep it conversational and concise.

User question: {user_input}

Relevant data:
{data_response}
"""

    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )

    reply = response.text.strip()

    # Optional: trim overly verbose replies
    if len(reply.split()) > 100:
        reply = reply.split('\n')[0]

    return reply, data_response
