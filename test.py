from transformers import AutoTokenizer

# Load the tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

# Encode the word 'I'
token_id = tokenizer.encode('I', add_special_tokens=False)

print(tokenizer.pad_token_id)

token_id2 = tokenizer.decode(token_id, add_special_tokens=False)

print(token_id2)

# Print the token ID
print(f"Token ID for 'I': {token_id}")

