#!/usr/bin/env python3
"""
Test script to find the correct OpenAI embedding model name.
"""
import openai
import sys
import os

# Test API key - must be set as environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

# Set the API key
openai.api_key = API_KEY

# Test text
test_text = "This is a test text for embedding generation."

# Model names to test
models_to_test = [
    "text-embedding-ada-002",      # Current model
    "text-embedding-3-small",      # Newer model
    "text-embedding-3-large",      # Newer model
]

print("Testing OpenAI embedding models...\n")

for model in models_to_test:
    try:
        print(f"Testing model: {model}")
        response = openai.embeddings.create(
            model=model,
            input=test_text
        )
        
        embedding = response.data[0].embedding
        print(f"✓ Success! Model '{model}' works.")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print()
        
    except Exception as e:
        print(f"✗ Failed! Model '{model}' error: {str(e)}")
        print()

print("\nTesting complete!")
