import os
import json

# Initialize the language model

# Function to load JSON safely
def load_json(filename):
    pth = os.getcwd()
    filepath = os.path.join(pth, "generations", "LLM_PC_attack_baseline", filename)

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        print(f"File not found: {filepath}")
        return None

# Load data from JSON files
data1 = load_json("good.json")  # Load first 5 entries
data2 = load_json("bad.json")  # Load first 5 entries

print("g", len(data1))

print("b",len( data2))

