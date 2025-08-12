#tool_weather.py
from urllib.parse import urljoin
import random
import json
from openai import OpenAI

# The "anyscale service deploy" script outputs a line that looks like
# 
#     curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
# 
# From this, you can parse out the service token and base URL.
token = <SERVICE_TOKEN>  # Fill this in. If deploying and querying locally, use token = "FAKE_KEY"
base_url = <BASE_URL>  # Fill this in. If deploying and querying locally, use base_url = "http://localhost:8000"

client = OpenAI(base_url= urljoin(base_url, "v1"), api_key=token)

# Dummy APIs
def get_current_temperature(location: str, unit: str = "celsius"):
    temperature = random.randint(15, 30) if unit == "celsius" else random.randint(59, 86) # Else Fahrenheit
    return {
        "temperature": temperature,
        "location": location,
        "unit": unit
    }

def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    temperature = random.randint(15, 30) if unit == "celsius" else random.randint(59, 86) # Else Fahrenheit
    return {
        "temperature": temperature,
        "location": location,
        "date": date,
        "unit": unit
    }

# Tools schema definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format \"City, State, Country\"."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit to return the temperature in. Defaults to \"celsius\"."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format \"City, State, Country\"."
                    },
                    "date": {
                        "type": "string",
                        "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit to return the temperature in. Defaults to \"celsius\"."
                    }
                },
                "required": ["location", "date"]
            }
        }
    }
]


# prettify printing the tools like this: "function_name(param1: type, param2: type, ...)"
tools_signature = []
for tool in tools:
    fn = tool["function"]
    fn_name = fn["name"]
    fn_args = fn["parameters"]["properties"]
    # args_signature = "param1: type, param2: type, ..."
    args_signature = ", ".join(f"{arg_name}: {arg['type']}" for arg_name, arg in fn_args.items())
    tools_signature.append(f"{fn_name}({args_signature})")
system_prompt =  "You are a weather assistant. Use the given functions to get weather data and provide the results."
user_prompt = "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2025-07-29."
print(f"== Tools defined:")
for tool in tools_signature:
    print(f"==== - {tool}")
print("== System Prompt:", system_prompt)
print("== User Prompt:", user_prompt)

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]
response = client.chat.completions.create(
    model="my-llama",
    messages=messages,
    tools=tools,
    tool_choice= "auto"
)

# Print the response tools
print("\n== Tool calls requested by the model:")
for tool_call in response.choices[0].message.tool_calls:
    print(f"==== - {tool_call.function.name}({tool_call.function.arguments})")


# Helper tool map (str -> python callable to your APIs)
helper_tool_map = {
    "get_current_temperature": get_current_temperature,
    "get_temperature_date": get_temperature_date
}

print("\n== Calling dummy APIs with the model's requests...")
# `response` is your model's last response containing the tool calls it requests.
for tool_call in response.choices[0].message.tool_calls:
    fn_call = tool_call.function
    
    fn_callable = helper_tool_map[fn_call.name]
    fn_args = json.loads(fn_call.arguments)

    output = json.dumps(fn_callable(**fn_args))

    # Create a new message of role `"tool"` containing the output of your tool
    messages.append({
        "role": "tool",
        "content": output
    })

print("\n== Sending the tool output back to the model...")
response = client.chat.completions.create(
    model="my-llama",
    messages=messages,
    stream=True
)

print("\nModel's final response:\n")
# Stream and print JSON
for chunk in response:
    data = chunk.choices[0].delta.content
    if data:
        print(data, end="", flush=True)