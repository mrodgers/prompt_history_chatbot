## Prompt History Chatbot Demo

This project demonstrates a versatile chatbot application that integrates multiple language models and APIs to provide users with a unique chat experience and prompt history exploration. It utilizes APIs from OpenAI, Anthropic, Google Vertex AI, and PromptMule, as well as local language models like Samantha, Llama, and Mistral.

### Features

- **Multi-API Integration**: The chatbot interfaces with various APIs, including OpenAI's GPT-3, Anthropic's Claude 2.1, Google's Vertex AI, and PromptMule.
- **Local Language Models**: Supports local models like Samantha, Llama, and Mistral for diverse response generation.
- **Prompt History Exploration**: Users can explore prompt histories based on date ranges and retrieve similar prompts based on semantic similarity.
- **Material Design UI**: The application leverages Panel's Material Design template for an elegant and responsive user interface.

### Setup

1. **Environment Setup**: Ensure Python 3.8 or higher is installed.
2. **Dependencies**: Install required Python packages listed in `requirements.txt`.
3. **API Keys**: You will need API keys from OpenAI, Anthropic, Google, and PromptMule.
4. **Google Authentication**: For Google Vertex AI, ensure you're logged in to the Google API.
5. **Running the Application**: Use the provided `run.sh` script to start the chatbot server.

### Usage

- **Chat Interface**: Interact with various language models through the chat interface.
- **Prompt History**: Use the date range picker to explore the history of prompts.
- **Similar Prompts**: Search for similar prompts based on user input and semantic score.

### Installation

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Run `pip install -r requirements.txt` to install dependencies.
4. Set up your `.env` file with the necessary API keys and configurations.
5. Run the application using `bash run.sh`.

### Important Scripts

- **chatbot.py**: The main Python script powering the chatbot application.
- **run.sh**: A Bash script to verify Python version, install dependencies, and run the server.

### Configuration

- **.env File**: Store your API keys and configurations in this file.
- **MODEL_ARGUMENTS**: Configure local language models in the Python script.

### Dependencies

see: requirements.txt

- `panel`
- `datetime`
- `ctranformers`
- `openai`
- `anthropic`
- `vertexai`
- `csv`
- `os`
- `dotenv`
- `requests`
- `json`

### Contact

For any queries or contributions, please reach out to the repository maintainers or open an issue in the project's GitHub repository.
