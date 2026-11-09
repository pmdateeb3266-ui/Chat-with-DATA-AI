"""Smoke check that a local Ollama server answers. Run: python scripts/check_ollama.py"""
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain data analysis in one sentence."
        }
    ]
)

print(response["message"]["content"])
