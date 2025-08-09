from enum import Enum
from urllib.parse import urljoin
from openai import OpenAI
from pydantic import BaseModel

# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
token = "<SERVICE_TOKEN>"  # Fill this in. If deploying and querying locally, use token = "FAKE_KEY"
base_url = "<BASE_URL>"  # Fill this in. If deploying and querying locally, use base_url = "http://localhost:8000"

client = OpenAI(base_url=urljoin(base_url, "v1"), api_key=token)

# Schema definition
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

# Query the model with enforced schema
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    # Set a `response_format` of type `json_schema` and define the schema there
    response_format= {
        "type": "json_schema",
        # Provide both `name`and `schema` (required)
        "json_schema": {
            "name": "car-description",
            "schema": CarDescription.model_json_schema() # Convert the pydantic model to a JSON schema
        },
    },
    stream=True
)

# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)
