import os
import json
from flask import Flask, request, jsonify
import openai

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Set it as an environment variable.")
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_dir, 'data.json')

with open(data_path, 'r') as f:
    data = json.load(f)

def gpt_4o_search(query, data):
    """
    This function simulates searching for a match in the data.
    You can either perform a simple lookup or integrate a call to the GPT‑4o API.
    """
    # Option 1: Simple lookup by key
    # if query in data:
    #     return data[query]
    
    # Option 2: Use the OpenAI API (using the new interface)
    prompt = (
        f"You are given the following data: {json.dumps(data)}\n"
        f"Select the text that best matches the query: '{query}'. "
        "If no text matches, respond with 'No matching text found.'"
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

@app.route('/data', methods=['POST'])
def data_request():
    req_data = request.get_json()
    if not req_data or 'query' not in req_data:
        return jsonify({"error": "No query provided"}), 400

    query = req_data['query']
    response_text = gpt_4o_search(query, data)
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
