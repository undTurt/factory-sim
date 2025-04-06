import os
import google.generativeai as genai
import sys

# Get API key from environment variable
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable is not set.")
    print("Please set it using:")
    print("PowerShell: $env:GOOGLE_API_KEY='your-api-key'")
    sys.exit(1)

try:
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring API: {str(e)}")
    sys.exit(1)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

# Add safety settings if needed
safety_settings = {
    "harassment": "block_none",
    "hate_speech": "block_none",
    "sexually_explicit": "block_none",
    "dangerous_content": "block_none"
}

def chat():
    print("Supply Chain Operations ChatBot (Type 'quit' to exit)")
    print("-" * 50)
    
    # Initialize chat session
    chat_session = model.start_chat(history=[])
    
    # Set initial system prompt
    system_prompt = """You are a specialized manufacturing consultant focused on electric motor production lines.
Your expertise includes:
- Identifying common assembly line bottlenecks and inefficiencies
- Suggesting process improvements for electric motor manufacturing
- Recommending lean manufacturing solutions
- Optimizing production flow and workstation layout
- Reducing assembly errors and quality issues
- Implementing error-proofing (poka-yoke) solutions
- Improving material handling and component staging
- Balancing production lines for optimal throughput

When analyzing problems:
1. Identify specific assembly line issues
2. Propose practical solutions with implementation steps
3. Consider cost-effectiveness and ease of implementation
4. Reference industry best practices from automotive and electronics manufacturing
5. Focus on reducing cycle time and improving first-pass yield

Use metrics like:
- Takt time
- Line balancing efficiency
- First-pass yield
- Assembly cycle time
- Throughput rate"""
    
    # Send system prompt
    _ = chat_session.send_message(system_prompt)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            # Get and print response
            response = chat_session.send_message(user_input)
            print("\nChatbot:", response.text)
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat()