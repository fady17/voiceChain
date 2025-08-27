# test_llm.py

from llama_cpp import Llama

# 1. Load the model
# The n_gpu_layers parameter is crucial.
# -1 means "offload all possible layers to the GPU".
# For the M4, this is what we want.
# Set verbose=True to see detailed logs on startup.
print("Loading model...")
llm = Llama(
    model_path="./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    n_gpu_layers=-1, 
    n_ctx=2048, # Context window size
    verbose=True
)
print("Model loaded successfully.")

# 2. Define the prompt
# We will use the Llama 3 chat template for optimal results.
# This format tells the model about the conversation structure.
prompt_messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant specialized in system architecture.",
    },
    {
        "role": "user",
        "content": "In simple terms, what is the key architectural advantage of Apple's Unified Memory for AI workloads?",
    },
]

# 3. Run inference
print("Generating response...")
output = llm.create_chat_completion(
    messages=prompt_messages,
    max_tokens=256  # Limit the response length
)

# 4. Print the output
# The response is in a dictionary structure.
response_text = output['choices'][0]['message']['content']
print("\n--- LLM Response ---")
print(response_text)
print("--------------------")