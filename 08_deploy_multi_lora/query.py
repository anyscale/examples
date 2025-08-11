#query.py
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

# Base model request (no adapter)
print("=== Base model ===")
response = client.chat.completions.create(
    model="my-llama",  # no adapter
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# nemoguard adapter (moderation)
print("=== LoRA: nemoguard ===")
# As per Nemoguard's usage instruction, add this to your system prompt
# https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control#system-instruction
TOPIC_SAFETY_OUTPUT_RESTRICTION = 'If any of the above conditions are violated, please respond with "off-topic". Otherwise, respond with "on-topic". You must respond with "on-topic" or "off-topic".'
messages_nemoguard = [
    {
        "role": "system",
        "content": f'In the next conversation always use a polite tone and do not engage in any talk about travelling and touristic destinations.{TOPIC_SAFETY_OUTPUT_RESTRICTION}',
    },
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello! How can I help today?"},
    {"role": "user", "content": "Do you know which is the most popular beach in Barcelona?"},
]
response = client.chat.completions.create(
    model="my-llama:nemoguard", ### with nemoguard adapter
    messages=messages_nemoguard,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# cv_job_matching adapter (structured JSON output)
print("=== LoRA: cv_job_matching ===")
messages_cv = [
    {
        "role": "system",
        "content": """You are an advanced AI model designed to analyze the compatibility between a CV and a job description. You will receive a CV and a job description. Your task is to output a structured JSON format that includes the following:

1. matching_analysis: Analyze the CV against the job description to identify key strengths and gaps.
2. description: Summarize the relevance of the CV to the job description in a few concise sentences.
3. score: Provide a numerical compatibility score (0-100) based on qualifications, skills, and experience.
4. recommendation: Suggest actions for the candidate to improve their match or readiness for the role.

Your output must be in JSON format as follows:
{
  "matching_analysis": "Your detailed analysis here.",
  "description": "A brief summary here.",
  "score": 85,
  "recommendation": "Your suggestions here."
}
""",
    },
    {
        "role": "user",
        "content": "<CV> Software engineer with 5 years of experience in Python and cloud infrastructure. </CV>\n<job_description> Looking for a backend engineer with Python and AWS experience. </job_description>",
    },
]
response = client.chat.completions.create(
    model="my-llama:cv_job_matching", ### with cv_job_matching adapter
    messages=messages_cv,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# yara adapter (cybersecurity task)
print("=== LoRA: yara ===")
messages_yara = [{"role": "user", "content": "Generate a YARA rule to detect a PowerShell-based keylogger. Generate ONLY the YARA rule, do not add explanations."}]
response = client.chat.completions.create(
    model="my-llama:yara", ### with yara adapter
    messages=messages_yara,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
