#query_all.py
from urllib.parse import urljoin
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum


# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
token = "<SERVICE_TOKEN>"  # Fill this in. If deploying and querying locally, use token = "FAKE_KEY"
base_url = "<BASE_URL>"  # Fill this in. If deploying and querying locally, use base_url = "http://localhost:8000"

client = OpenAI(base_url=urljoin(base_url, "v1"), api_key=token)



print("====== JSON SCHEMA Example (response_format.type=json_schema) ===\n")

# 1. Define your schema using a Pydantic model
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

user_prompt = "Generate a JSON of the most iconic car from the 90's."
print(f"User prompt: {user_prompt}")
# 2. Make a call with the JSON schema
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {
            "role": "user",
            "content": user_prompt,
        }
    ],
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

print("Response:\n")
# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)


print("\n\n")
print("====== JSON SCHEMA Example (extra_body.guided_json) ===\n")


# 1. Define your schema using a Pydantic model
# See last example for the schema definition

#user_prompt = # See last example for the user prompt
print(f"User prompt: {user_prompt}")
# 2. Make a call with the JSON schema
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {
            "role": "user",
            "content": user_prompt,
        }
    ],
    # 3. Pass it to the `guided_json` field as an `extra_body` parameter
    extra_body={
        "guided_json": CarDescription.model_json_schema() # Convert the pydantic model to a JSON schema
    }, 
    stream=True
)

print("Response:\n")
# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)


print("\n\n")
print("====== non-enforced JSON SCHEMA Example (response_format.type=json_object) ===\n")

#user_prompt = # See last example for the user prompt
print(f"User prompt: {user_prompt}")
response = client.chat.completions.create(
    model="my-qwen",
    # Set the type of `response_format` to `json_object`
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": user_prompt + "Limit to three fields.",
        }
    ],
    stream=True
)

# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)


print("\n\n")
print("====== Guided Choice Example ===\n")

# 1. Define the valid choices
choices = ["Purple", "Cyan", "Magenta"]

print(f"Choices (not given in any prompt): {choices}")
user_prompt = "Pick a color"
print(f"User prompt: {user_prompt}")
# 2. Make a call with the choices
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Always reply with one of the choices provided"},
        {"role": "user", "content": user_prompt}
    ],
    # 3. Pass it to the `guided_choice` field as an `extra_body`
    extra_body={
        "guided_choice": choices
    },
    stream=True
)

print("Response:\n")
# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)


print("\n\n")
print("====== Guided Regex Example ===\n")


# 1. Define a regex pattern for a hex color code
email_pattern = r"^customprefix\.[a-zA-Z]+@[a-zA-Z]+\.com\n$" 

print(f"Regex pattern (not provided in any prompt): {email_pattern}")
user_prompt = "Generate an example email address for Alan Turing, who works in Enigma. End your answer with a new line"
# 2. Make a call with the regex pattern
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Always reply following the pattern provided"},
        {"role": "user", "content": user_prompt}
    ],
    # 3. Pass it to the `guided_regex` field as an `extra_body`
    # (Optional) For more reliability, add a `stop` parameter and include it at the end of your pattern
    extra_body={
        "guided_regex": email_pattern,
        "stop": ["\n"]
    },
    stream=True
)

# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)


print("\n\n")
print("====== Guided Grammar Example ===\n")

# 1. Define the grammar
simplified_sql_grammar = """
start: "SELECT " columns " from " table ";"

columns: column (", " column)?
column: "username" | "email" | "*"

table: "users"
"""

print(f"Grammar (not provided in any prompt): {simplified_sql_grammar}")
user_prompt = "Generate an SQL query to show the 'username' and 'email' from the 'users' table."
print(f"User prompt: {user_prompt}")
# 2. Make a call with the grammar
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {"role": "system", "content": "Respond with a SQL query using the grammar."},
        {"role": "user", "content": user_prompt}
    ],
    # 3. Pass it to the `guided_grammar` field as an `extra_body`
    extra_body={
        "guided_grammar": simplified_sql_grammar
    },
    stream=True
)

print("Response:\n")
# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)



print("\n\n")
print("====== Structural Tags Example ===\n")


# 1. Describe the overall structural constraint in a system prompt
system_prompt = """
You are a helpful assistant.

You can answer user questions and optionally call a function if needed. If calling a function, use the format:
<function=function_name>{"arg1": value1, ...}</function>

Example:
<function=get_weather>{"city": "San Francisco"}</function>

Task:
Start by writing a short description (two sentences max) of the requested city.
Then output a form that uses the tags below, one tag per line.
Finish by writing a short conclusion (two sentences max) on the main touristic things to do there.

Required tag blocks
<city-name>{"value": string}</city-name>
<state>{"value": string}</state>
<main-borough>{"value": string}</main-borough>
<baseball-teams>{"value": [string]}</baseball-teams>
<weather>{"value": string}</weather>
"""

print("Structures schema and triggers not defined in any prompt. See code for details...")
user_prompt = "Tell me about a city in the east coast of the U.S"
print(f"User prompt: {user_prompt}")
# 2. Define the structural rules to follow (one per field)
structures = [
    {  # <city-name>{"value": "Boston"}
        "begin": "<city-name>",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "end": "</city-name>",
    },
    {  # <state>{"value": "MA"}
        "begin": "<state>",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "end": "</state>",
    },
    {  # <main-borough>{"value": "Charlestown"}
        "begin": "<main-borough>",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        "end": "</main-borough>",
    },
    {  # <baseball-teams>{"value": ["Red Sox"]}
        "begin": "<baseball-teams>",
        "schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["value"],
        },
        "end": "</baseball-teams>",
    },
    {
        "begin": "<weather>",
        "schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "pattern": r"^<function=get_weather>\{.*\}</function>$"
                }
            },
            "required": ["value"],
        },
        "end": "</weather>",
    },
    {
        "begin": "<function=get_weather>",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        },
        "end": "</function>"
    }
]

# 3. Define the trigger(s): whenever the model types "<city-name" etc.,
triggers = ["<city-name", "<state", "<main-borough", "<baseball-teams", "<weather", "<function="]

# 4. Make a call with the structures and triggers
response = client.chat.completions.create(
    model="my-qwen",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt
        },
    ],
    #
    response_format={
        "type": "structural_tag",
        "structures": structures,
        "triggers": triggers,
    },
    stream=True
)

print("Response:\n")
# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)