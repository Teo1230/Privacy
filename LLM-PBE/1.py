from transformers import pipeline

# Initialize the pipeline
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-70B")

# Generate text based on a query
query = "Your input text here"
result = pipe(query, max_length=100)  # Adjust max_length as needed

# Print the generated text
print(result)
