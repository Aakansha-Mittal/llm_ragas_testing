from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

try:
    response = client.chat.completions.create(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one sentence."
            }
        ]
    )

    print("\nOLLAMA SUCCESS")
    print(response.choices[0].message.content)

except Exception as e:
    print("\nOLLAMA ERROR")
    print(type(e).__name__)
    print(e)