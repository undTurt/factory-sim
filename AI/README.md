# Supply Chain Operations ChatBot

A chatbot powered by Google's Gemini-1.5-pro model, specialized in assembly line and supply chain operations.

## Prerequisites

- Python 3.7 or higher
- Google API key for Gemini

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Google API key:
   - Option 1: Set environment variable:
     ```bash
     export GOOGLE_API_KEY='your-api-key'  # For Unix/Mac
     set GOOGLE_API_KEY='your-api-key'     # For Windows
     ```
   - Option 2: Replace the hardcoded key in the script (not recommended for public repositories)

## Usage

Run the chatbot:
```bash
python chatbotgemi.py
```

Type 'quit' or 'exit' to end the conversation.

## Security Note

Make sure to never commit your actual API key to the repository. If you accidentally committed it, you should reset it immediately.