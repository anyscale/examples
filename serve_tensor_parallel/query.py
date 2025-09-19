import os
from urllib.parse import urljoin
import requests

# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
token = <SERVICE_TOKEN>  # Fill this in. If deploying and querying locally, use token = "FAKE_KEY"
base_url = <BASE_URL>  # Fill this in. If deploying and querying locally, use base_url = "http://localhost:8000"

resp = requests.get(
    urljoin(base_url, "infer"),
    params={"text": "What is the future of AI? "},
    headers={"Authorization": f"Bearer {token}"})

print(resp.text)