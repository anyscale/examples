from urllib.parse import urljoin
from openai import OpenAI

# The "anyscale service deploy" command outputs a line that looks like
#
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
#
# From this, you can parse out the service token and base URL.
token = "<SERVICE_TOKEN>"
base_url = "<BASE_URL>"

client = OpenAI(base_url=urljoin(base_url, "v1"), api_key=token)

response = client.chat.completions.create(
    model="my-model",
    messages=[
        {"role": "user", "content": "What's the capital of France?"}
    ],
    stream=True,
)

for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)
print()
