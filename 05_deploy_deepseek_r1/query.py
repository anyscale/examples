from urllib.parse import urljoin
from openai import OpenAI

# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
token = <SERVICE_TOKEN>  # Fill this in. If deploying and querying locally, use token = "FAKE_KEY"
base_url = <BASE_URL>  # Fill this in. If deploying and querying locally, use base_url = "http://localhost:8000"

client = OpenAI(base_url= urljoin(base_url, "v1"), api_key=token)

response = client.chat.completions.create(
    model="my-deepseek-r1",
    messages=[
        {"role": "user", "content": "What's the capital of France?"}
    ],
    stream=True
)

# Stream and print JSON
for chunk in response:
    # Stream reasoning content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        data_reasoning = chunk.choices[0].delta.reasoning_content
        if data_reasoning:
            print(data_reasoning, end="", flush=True)
    # Later, stream the final answer
    if hasattr(chunk.choices[0].delta, "content"):
        data_content = chunk.choices[0].delta.content
        if data_content:
            print(data_content, end="", flush=True)