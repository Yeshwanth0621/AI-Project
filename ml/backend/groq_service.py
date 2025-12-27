import os
from groq import Groq

# Initialize Groq client
# Ensure GROQ_API_KEY is set in environment variables
api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the application.")

client = Groq(api_key=api_key)

def stream_chat(message):
    """
    Generates a streaming response from Groq API.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a futuristic AI assistant. Your interface is high-tech and minimal. Keep your responses concise, intelligent, and helpful."
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        yield f"Error: {str(e)}"
