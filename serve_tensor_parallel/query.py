import os
from urllib.parse import urljoin
import requests

# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
# From this, you can parse out the service token and base URL.
token = "eWoFaAMR9SN4KnCXE-ExE6lvZG4lbTN6IXzkvJFB61A"
base_url = "https://tp-service-jgz99.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/"

token = "ca21hjrgCgQuZUTERmc9gzvxBW880IR1ivrY1cqrgd0"
base_url = "https://serve-session-raxdx23jzvfjvc3if1b5wnma2t.i.anyscaleuserdata.com/"

resp = requests.get(
    urljoin(base_url, "infer"),
    params={"text": "What is the future of AI? "},
    headers={"Authorization": f"Bearer {token}"})

print(resp.text)