import os
import json
import numpy as np
from models.ft_clm import PeftCasualLM, FinetunedCasualLM

# Initialize the language model
llm = FinetunedCasualLM(
    model_path="LLM-PBE/Llama3.1-8b-instruct-LLMPC-Red-Team", 
    arch="LLM-PBE/Llama3.1-8b-instruct-LLMPC-Red-Team", 
    max_seq_len=16
)

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
data1 = load_json("good.json")[:517]  # Load first 5 entries
data2 = load_json("bad.json")[:517]    # Load first 5 entries

# Process and save results for data1
for i in data1:
    try:
        prompt = i['prompt']
#        res = llm.query(prompt, new_str_only=True)
        
        # Get the last token embedding
        last_token_embedding = llm.get_all_layers_embeddings_with_cache(prompt)
        print(last_token_embedding)
        # Convert ndarray to a list before storing
        i['emb'] = last_token_embedding.tolist() if isinstance(last_token_embedding, np.ndarray) else last_token_embedding

    except Exception as e:
        print(f"ERROR at {i}-th prompt: {prompt}\n", e)
        continue

# Save data1 with embeddings
with open("good_emb_all_layers.json", 'w', encoding='utf-8') as f:
    json.dump(data1, f, ensure_ascii=False, indent=4)

# Process and save results for data2
for i in data2:
    try:
        prompt = i['prompt']
 #       res = llm.query(prompt, new_str_only=True)
        
        # Get the last token embedding
        last_token_embedding = llm.get_all_layers_embeddings_with_cache(prompt)

        # Convert ndarray to a list before storing
        i['emb'] = last_token_embedding.tolist() if isinstance(last_token_embedding, np.ndarray) else last_token_embedding

    except Exception as e:
        print(f"ERROR at {i}-th prompt: {prompt}\n", e)
        continue

# Save data2 with embeddings
with open("bad_emb_all_layers.json", 'w', encoding='utf-8') as f:
    json.dump(data2, f, ensure_ascii=False, indent=4)
print("da")
