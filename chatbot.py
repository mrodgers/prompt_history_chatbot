import panel as pn
import datetime as dt
from panel.widgets import DatetimeRangeInput
from ctransformers import AutoModelForCausalLM
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import time
import csv
import os
from dotenv import load_dotenv
import requests
import json

# Load the environment variables from .env file
load_dotenv()

# Accessing the API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

# Leverage the Material Design template
pn.extension(template='material')
pn.state.template.param.update(title="Prompt History Chatbot Demo")

# Google Vertex API, this is the google project you are using, also you will need to be logged in to google api
# pip3 install --upgrade --user google-cloud-aiplatform
# gcloud auth application-default login
google_project_id = os.getenv('GOOGLE_PROJECT_ID')
google_location_id = os.getenv('GOOGLE_LOCATION_ID')

# get PromptMule's API key set up
promptmule_api_key =os.getenv('PROMPTMULE_API_KEY')

# this is the file where responses are saved along side of each other, with latencies
response_filename = 'multi_llm_responses.csv'

# After loading the environment variables
print("OpenAI API Key:", openai_api_key)
print("Anthropic API Key:", anthropic_api_key)
print("Google Setup: ", google_location_id, google_project_id)
print("PromptMule API Key: ", promptmule_api_key)

# Model arguments for local language models
MODEL_ARGUMENTS = {
    # Define each model's arguments and keyword arguments
    "samantha": {
        "args": ["TheBloke/Dr_Samantha-7B-GGUF"],
        "kwargs": {"model_file": "dr_samantha-7b.Q5_K_M.gguf"},
    },
    "llama": {
        "args": ["TheBloke/Llama-2-7b-Chat-GGUF"],
        "kwargs": {"model_file": "llama-2-7b-chat.Q5_K_M.gguf"},
    },
    "mistral": {
        "args": ["TheBloke/Mistral-7B-Instruct-v0.1-GGUF"],
        "kwargs": {"model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"},
    },
}

# Function to generate text via Google Vertex AI
def generate_text_with_vertex_ai(project_id: str, location: str, query: str) -> str:
    if not project_id:
        print("Skipping Vertex AI due to missing Google Project ID.")
        return "Skipping Vertex AI due to missing Google Project ID."
    if not location:
        print("Skipping Vertex AI due to missing Google Project Location.")
        return "Skipping Vertex AI due to missing Google Project Location."
    
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    
    print("Sending prompt to: Vertex AI")

    # Query the model
    response = multimodal_model.generate_content(
        [
            # Add your query here
            Part.from_text(query)
        ]
    )
    return response.text

# Function to write to CSV

# Function to generate text via PromptMule's proxy cache to OpenAI
def generate_text_with_promptmule(promptmule_api_key, contents):
    if not promptmule_api_key:
        print("Skipping PromptMule due to missing API key.")
        return "PromptMule API key not provided, get one at https://app.promptmule.com"

    # API endpoint - replace with the actual PromptMule API endpoint
    url = 'https://api.promptmule.com/prompt'

    # Headers with API Key for authentication
    headers = {
        'x-api-key': promptmule_api_key,
        'Content-Type': 'application/json'
    }

    # Data to be sent in the request
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": contents  # Corrected from {contents} to contents
            }
        ],
        "max_tokens": "512",        # Corrected to integer
        "temperature": "0",         # Corrected to integer or float
        "api": "openai",
        "semantic": "0.99",         # Corrected to float
        "sem_num": "2"              # Corrected to integer
    }

    print("Sending prompt to: PromptMule")

    # Make a POST request to the API
    response = requests.post(url, json=data, headers=headers)

    response_json = response.json()
    promptmule_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    promptmule_cache = response_json.get("choices", [{}])[0].get("is_cached")

    # Check if the request was successful
    if response.status_code == 200:
        # Return the response content
        return promptmule_cache, promptmule_response
    else:
        # Handle errors (you can expand this part based on your error handling policy)
        return f'Error: {response.status_code}, {response.text}'

# Function to generate text using Anthropic's Claude 2.1
def generate_text_with_anthropic(api_key, prompt):
    """
    Generates text using Anthropic's Claude 2.1.

    :param prompt: The prompt to send to the model :return: The generated text.
    """
    if not api_key:
        print("Skipping Anthropic due to missing API key.")
        return "Anthropic API key not provided."
    
    # Setting up Anthropic API
    print("Sending prompt to: Anthropic")
    anthropic = Anthropic(api_key=api_key)
    
    # Use the API keys in your application
    # For example, setting the Anthropic API key
    try:
        # Call the OpenAI API
        response = anthropic.completions.create(
            model="claude-2.1",
            max_tokens_to_sample=512,
            temperature=0,
            prompt=f"{HUMAN_PROMPT} {prompt}\n{AI_PROMPT}"
        )
        # Return the generated text
        return response

    except Exception as e:
        return f"An error occurred: {e}"

# Function to generate text using OpenAI's GPT-3
def generate_text_with_openai(api_key, prompt):
    """
    Generates text using OpenAI's GPT-3.

    :param prompt: The prompt to send to the model :return: The generated text.
    """
    if not api_key:
        print("Skipping OpenAI due to missing API key.")
        return "OpenAI API key not provided, get one at https://openai.com"

    # Use the API keys in your application
    # For example, setting the OpenAI API key
    openai.api_key = api_key
    print("Sending prompt to: OpenAI")

    try:
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or another model like "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512
        )
        # Return the generated text
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

# Write the gathered data to CSV
def write_to_csv(prompt, responses, latencies, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is being created
        if not file_exists:
            headers = ['Prompt'] + [f'Response from LLM {i+1}' for i in range(len(responses))] + [f'Latency for LLM {i+1}' for i in range(len(latencies))]
            writer.writerow(headers)

        # Write the data
        writer.writerow([prompt] + responses + latencies)



# Callback function to handle date range input
def create_on_date_range_change(token, api_key, prompt_display_pane, quantity_select, prompt_count_pane):
    def on_date_range_change(event):
        # When called directly, event is a dictionary with 'new' key
        # When called by widget event, event is an object with 'new' attribute
        if isinstance(event, dict):
            start_date, end_date = event['new']
        else:
            start_date, end_date = event.new

        print("Selected Date Range:", start_date, "to", end_date)

        # Format dates for API call
        formatted_start_date = start_date.strftime('%Y-%m-%d')
        formatted_end_date = end_date.strftime('%Y-%m-%d')

        num_prompts = quantity_select.value
        # Fetch recent prompts based on the selected date range
        recent_prompts = get_date_based_prompt_history(token, api_key, formatted_start_date, formatted_end_date, num_prompts)
        if not recent_prompts:
            print("Retrieved Prompt History with Zero Prompt Results")
            # Create a single prompt entry with a placeholder text
            recent_prompts = [{'request-time': formatted_start_date, 'prompt': 'Range Empty for Prompt History'}]
        else:
            print("Retrieved Prompt History.")

        # Update prompt display pane
        prompt_display = '\n\n'.join([f"{p['request-time']}: {p['prompt']}" for p in recent_prompts])
        prompt_display_pane.object = prompt_display
      
        # Count the actual number of prompts retrieved
        actual_count = len(recent_prompts)
        prompt_count_pane.object = (f"Ask: {num_prompts} Retrieved: {actual_count}")

    return on_date_range_change

def date_based_prompt_history(token, api_key, start_date, end_date):
    url = f'https://api.promptmule.com/prompt?start-date={start_date}&end-date={end_date}'

    headers = {
        'x-api-key': api_key,
        'Authorization': f'Bearer {token}'
    }

    try:
        #print(f'Calling promptmule with: {url} {headers}')
        response = requests.get(url, headers=headers, verify=True)  # SSL verification enabled
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        return f'HTTP error occurred: {http_err} - {response.text}'
    except Exception as err:
        return f'Other error occurred: {err}'

def login_promptmule(username, password):
    url = 'https://api.promptmule.com/login'
    headers = {'Content-Type': 'application/json'}
    data = {
        "username": username,
        "password": password
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get('success'):
            return response_data.get('token')
        else:
            return response_data.get('message')
    except requests.RequestException as e:
        return f'Error during login: {e}'

def get_date_based_prompt_history(token, api_key, formatted_start_date, formatted_end_date, num_prompts=10):
    # Calculate date range: end date is today, start date is `num_prompts` days before
    end_date = formatted_end_date
    start_date = formatted_start_date

    try:
        response = date_based_prompt_history(token, api_key, str(start_date), str(end_date))

        # Check if the 'prompts' array is empty
        if not response['prompts']:
            print("Prompt history is empty.")
            return []
        else:
            print("Prompts retrieved.")

        # Check if response is a dictionary and contains a key 'prompts'
        if isinstance(response, dict) and 'prompts' in response:
            # Return the first `num_prompts` items from the 'prompts' list
            return response['prompts'][:num_prompts]
        else:
            # Log error or return empty list if the response is not as expected
            print("Unexpected response format or error:", response)
            return []

    except Exception as e:
        # Handle any exceptions that occurred during the API call
        print(f"An error occurred while fetching prompt history: {e}")
        return []

def get_similar_prompts(user_prompt, api_key, model, sem_score_max, num_prompts=10):
    """
    Retrieves prompts similar to the user-provided prompt based on a semantic score range.

    :param user_prompt: The prompt provided by the user.
    :param token: Authentication token for the API.
    :param api_key: API key for accessing PromptMule.
    :param sem_score_min: Minimum semantic score for similarity matching.
    :param sem_score_max: Maximum semantic score for similarity matching.
    :param num_prompts: Number of similar prompts to retrieve. Defaults to 10.
    :return: A list of similar prompts or an empty list in case of errors.
    """
    try:
        # Assuming a function `similar_prompts` exists in PromptMule's API to fetch similar prompts
        response = similar_prompts(user_prompt, api_key, model, sem_score_max, num_prompts)
        print("in get_similar_prompts(), response:", response)
        # Check if the response contains similar prompts
        if 'prompt' in response and response['prompt']:
            print("Similar prompts retrieved.")

            # Extracting the similar prompts and their semantic scores
            similar_prompts_list = response

            return similar_prompts_list
        else:
            print("No similar prompts found or empty response.")
            return []

    except Exception as e:
        # Handle any exceptions that occurred during the API call
        print(f"An error occurred while fetching similar prompts: {e}")
        return []

def search_similar_prompts(event):
    user_prompt = prompt_search_box.value
    if user_prompt:
        model = "gpt-3.5-turbo"  # Define the model here, or fetch from user/UI
        sem_score = str(sem_score_slider.value)  # Use the value from the slider
        sem_num = "10"  # Define or fetch the number of semantic matches

        similar_prompts_response = get_similar_prompts(user_prompt, promptmule_api_key, model, sem_score, sem_num)
        print("in search_similar_prompts(), response:", similar_prompts_response)

        if 'choices' in similar_prompts_response:
            display_text = "\n\n".join([
                f"Response: {choice['message']['content']} Similarity: {choice.get('score', 'N/A')}"
                for choice in similar_prompts_response['choices']
            ])
            prompt_display_pane.object = display_text
            prompt_count_pane.object = f"Retrieved {len(similar_prompts_response['choices'])} similar prompts."
        else:
            prompt_display_pane.object = "No similar prompts found."
            prompt_count_pane.object = f"Retrieved: 0"
    else:
        prompt_display_pane.object = "Enter a prompt for similarity search."
        prompt_count_pane.object = ""

def similar_prompts(user_prompt, api_key, model, sem_score, sem_num):
    """
    Retrieves similar prompts from PromptMule API based on semantic similarity.

    :param user_prompt: The prompt provided by the user.
    :param api_key: API key for accessing PromptMule.
    :param model: AI model to use for processing the prompt.
    :param sem_score: Semantic match percentage for cache retrieval.
    :param sem_num: Number of semantic matches to return.
    :return: The API response containing similar prompts or error message.
    """
    # API endpoint
    url = "https://api.promptmule.com/prompt"

    # Headers and payload for the API request
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": "500",
        "temperature": "0",
        "api": "openai",
        "semantic": sem_score,
        "sem_num": sem_num
    }

    print("payload: ", payload)
    
    # Executing the POST request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print("in similar_prompts(), response:", response)

        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

        # Parsing the JSON response
        return response.json()

    except requests.exceptions.HTTPError as err:
        return f"HTTP Error: {err}"
    except Exception as e:
        return f"Error: {e}"

# Callback function for the chat interface
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    responses = []
    latencies = []
    model_names = []
    
    # Define prompts for local LLMs
    alpaca_prompt = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {contents}

    ### Response:'''

    llama_prompt = f'''
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant.
    <</SYS>>
    {contents} [/INST]</s>'''

    mistral_prompt = f'''<s>[INST] Please respond to the Question : {contents} [/INST]'''
 
 
    # Code to interact with each API and local LLMs
    # Include checks for API keys before making calls
 
    # OpenAI response
    print("OpenAI Key: ", openai_api_key)
    start_time = time.time()
    openai_response = generate_text_with_openai(openai_api_key, contents)
    openai_latency = time.time() - start_time

    model_names.append("OpenAI")

    if openai_api_key:
        responses.append(openai_response)
        latencies.append(openai_latency)
        instance.stream(openai_response + '\nlatency: ' + f'{openai_latency:.3f}', user="OpenAI", message=None)
    else:
        responses.append("")
        latencies.append(0)
        instance.stream(openai_response, user="System", message=None)

 
    # Anthropic response
    print("Anthropic Key: ", anthropic_api_key)
    model_names.append("Anthropic")

    if anthropic_api_key:
        start_time = time.time()
        anthropic_response = generate_text_with_anthropic(anthropic_api_key, contents)
        anthropic_latency = time.time() - start_time
        responses.append(anthropic_response.completion)
        latencies.append(anthropic_latency)
        instance.stream(anthropic_response.completion + '\nlatency: ' + f'{anthropic_latency:.3f}', user="Anthropic", message=None)
    else:
        responses.append("")
        latencies.append(0)
        instance.stream("Anthropic API Key not found.", user="System", message=None)
 

    # Google Vertex response
    model_names.append("Google Vertex AI")

    if google_project_id:
        if google_location_id:
            start_time = time.time()
            vertex_ai_response = generate_text_with_vertex_ai(google_project_id, google_location_id, contents)
            vertex_ai_latency = time.time() - start_time
            responses.append(vertex_ai_response)
            latencies.append(vertex_ai_latency)
            instance.stream(vertex_ai_response+ '\nlatency: ' + f'{vertex_ai_latency:.3f}', user="Google Vertex AI", message=None)
        else:
            print("Missing Vertex AI: Google Project Location")
            responses.append("")
            latencies.append(0)
            instance.stream("Missing Vertex AI: Google Project Location", user="System", message=None)
    else:
        responses.append("")
        latencies.append(0)
        print("Missing Vertex AI: Google Project ID")
        instance.stream("Missing Vertex AI: Google Project ID", user="System", message=None)

            
    # PromptMule response
    model_names.append("PromptMule")
 
    if promptmule_api_key:
        start_time = time.time()
        promptmule_cache, promptmule_response = generate_text_with_promptmule( promptmule_api_key, contents)
        promptmule_latency = time.time() - start_time
        responses.append(promptmule_response)
        latencies.append(promptmule_latency)
        instance.stream(promptmule_response + '\nlatency: ' + f'{promptmule_latency:.3f}' + "\ncached: " + str(promptmule_cache), user="PromptMule", message=None)
    else:
        print("PromptMule API Key not set.")
        responses.append("")
        latencies.append(0)
        instance.stream("PromptMule API Key not set. Get one at https://app.promptmule.com", user="System", message=None)
 
    
    
    # delimit SaaS v. Local LLMs
 
    for model in MODEL_ARGUMENTS:
        try:
            # Check if the model is already loaded
            if model not in pn.state.cache:
                print(f"Loading model: {model}")
                pn.state.cache[model] = AutoModelForCausalLM.from_pretrained(
                    *MODEL_ARGUMENTS[model]["args"],
                    **MODEL_ARGUMENTS[model]["kwargs"],
                    gpu_layers=30,
                )

            llm = pn.state.cache[model]

            # Set the appropriate prompt based on the model
            if model == 'samantha':
                prompt = alpaca_prompt
            elif model == 'llama':
                prompt = llama_prompt
            elif model == 'mistral':
                prompt = mistral_prompt
            else:
                print(f"Warning: No specific prompt found for model {model}. Using a default prompt.")
                prompt = "Default prompt text"

            print("Sending prompt to: ", model)

            # Generate response and measure latency
            start_time = time.time()
            response = llm(prompt, max_new_tokens=512, stream=False)
            model_latency = time.time() - start_time

            # Stream the response
            message = None  # Placeholder for any additional processing
            instance.stream(response.strip() + '\nlatency: ' + f'{model_latency:.3f}', user=model.title(), message=message)

            # Append results to lists
            model_names.append(model)
            responses.append(response.strip())
            latencies.append(model_latency)

        except Exception as e:
            print(f"Error processing model {model}: {e}")
 
     
    # summarize latencies
    #latencies_string = ", ".join([f"{model}: {latency:.3f}" for model, latency in zip(model_names, latencies)])
    # Determine the maximum length of model names for formatting
    max_name_length = max(len(name) for name in model_names)

    # Header for the table
    #table_header = f"{'Model Name':<{max_name_length}} | Latency (s)\n" + "-" * (max_name_length + 15)
    # Rows of the table
    table_rows = "\n".join([f"{model:<{max_name_length}} was {latency:.3f}sec" for model, latency in zip(model_names, latencies)])

    # Final table string
    #latencies_string = table_header + "\n" + table_rows
    latencies_string = table_rows

    instance.stream(latencies_string, user="LLM Latency Report", message=message)

    
    # Write to CSV
    write_to_csv(contents, responses, latencies, response_filename)

# Initialize the prompt display pane with the new styles
prompt_display_pane = pn.pane.Markdown(
    name="Recent Prompts",
    styles={'background': 'white'}, 
    margin=(10, 10, 10, 10),
    css_classes=['box-shadow', 'scrollable-pane']  # Added 'scrollable-pane' class
)

# Initialize a pane to display the count of retrieved prompts
prompt_count_pane = pn.pane.Markdown("", name="Retrieved Count")

# Create a slider for semantic score selection
sem_score_slider = pn.widgets.FloatSlider(name='Similarity Percentage', start=0.1, end=1.0, step=0.01, value=0.95)

# Calculate default dates
default_start_date = dt.datetime.now() - dt.timedelta(days=7)
default_end_date = dt.datetime.now()
# Perform the login to get the token

username = os.getenv('PROMPTMULE_USERNAME')
password = os.getenv('PROMPTMULE_PASSWORD')
token = login_promptmule(username, password)

# Custom CSS to enhance the UI with Material Design principles and add scrollable prompt history pane
pn.config.raw_css.append('''
.material-pane {
    border-radius: 4px;  /* Rounded corners */
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* Elevation effect */
    padding: 16px;  /* Consistent padding */
    background-color: #FFFFFF;  /* Light background */
}

.material-label {
    font-weight: bold;
    margin-bottom: 16px;
}

.material-select {
    margin-bottom: 16px;
}

.material-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.scrollable-prompt-pane {
    max-height: 300px; /* Fixed maximum height */
    overflow-y: auto; /* Vertical scroll for overflow */
    overflow-x: auto; /* Hide horizontal scroll */
    white-space: normal; /* Allow text wrapping */
}

.fixed-width-prompt-history {
    width: 50%; /* Fixed width for prompt history layout */
}

/* iPhone screens */
@media (max-width: 414px) {
    .material-pane {
        /* styles for material-pane on iPhone screens */
    }
    .material-row {
        /* styles for material-row on iPhone screens */
    }
    .fixed-width-prompt-history {
        width: 100%; /* Full width on iPhone screens */
    }
    .datetime-picker-font {
        font-size: 8px; /* Smaller font size for iPhone screens */
    }
}

/* Smaller screens (e.g., tablets) */
@media (max-width: 768px) {
    .material-pane {
        /* styles for material-pane on smaller screens */
    }
    .material-row {
        /* styles for material-row on smaller screens */
    }
    .fixed-width-prompt-history {
        width: 100%; /* Full width on smaller screens */
    }
    .datetime-picker-font {
        font-size: 10px; /* Smaller font size for smaller screens */
    }
}

/* Medium screens (e.g., small laptops) */
@media (max-width: 1024px) {
    .material-pane {
        /* styles for material-pane on medium screens */
    }
    .material-row {
         /* styles for material-row on medium screens */
    }
    .fixed-width-prompt-history {
        width: 50%; /* Half width on medium screens */
    }
    .datetime-picker-font {
        font-size: 11px; /* Medium font size for medium screens */
    }
}

@media (max-width: 2048px) {
    .material-pane {
    }
    .material-row {
    }
    .fixed-width-prompt-history {
        width: 30%; /* Full width on larger screens */
    }
    .datetime-picker-font {
        font-size: 8px; /* Smaller font size for smaller screens */
    }
}
''')

# Initialize the Select widget for quantity selection
quantity_select = pn.widgets.Select(
    name='Prompts to Retrieve',
    options=[1, 10, 50, 100, 500, 1000],  # String options for dropdown
    value=10,  # Default value
    width=125  # Adjust width as needed
)

# Initialize the datetime range picker
values = (default_start_date, default_end_date)
datetime_range_picker = pn.widgets.DatetimeRangePicker(
    name='Prompt Creation Date (GMT)',
    value=values,
    margin=(10, 10, 10, 10),
    css_classes=['box-shadow', 'datetime-picker-font'],
    styles={'background': 'white'}
)

# Create on_date_range_change with the obtained token and include prompt_count_pane
on_date_range_change = create_on_date_range_change(token, promptmule_api_key, prompt_display_pane, quantity_select, prompt_count_pane)

# Watch the correct callback function
datetime_range_picker.param.watch(on_date_range_change, 'value')

# Fetch initial prompts based on the default date range
on_date_range_change({'new': values})

# Apply custom classes to widgets and panes
datetime_range_picker.css_classes = ['material-select']
quantity_select.css_classes = ['material-select']
prompt_display_pane.css_classes = ['material-pane', 'scrollable-prompt-pane']
prompt_count_pane.css_classes = ['material-label']

# Create a text input widget for the user prompt
prompt_search_box = pn.widgets.TextInput(name='Prompt Explorer', placeholder='Enter your prompt here to discover similar responses...')

# Create a button for initiating the search
search_button = pn.widgets.Button(name='Search', button_type='primary')
search_button.on_click(search_similar_prompts)

# # Use Column and Row for better layout
prompt_history_layout = pn.Column(
    pn.pane.Markdown("##  Search", css_classes=['widget-title']),
    pn.Row(search_button, css_classes=['material-row']),  # Added search box and button 
    pn.Row(prompt_search_box, css_classes=['material-row']),  # Added search box and button
    pn.Row(sem_score_slider, css_classes=['material-row']),  # Add the semantic score slider
    pn.Row(pn.pane.Markdown(css_classes=['material-label']), datetime_range_picker, css_classes=['material-row']),
    pn.Row(pn.pane.Markdown(css_classes=['material-label']), quantity_select, css_classes=['material-row']),
    prompt_display_pane,
    prompt_count_pane,
    css_classes=['material-pane', 'fixed-width-prompt-history'],  # Added class
    margin=(10, 10, 10, 10)
)

# Initialize the chat interface
chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send(
    "Send a message to get a response from each: PromptMule, Llama 2, Mistral (7B), Google, Anthropic and OpenAI!",
    user="System",
    respond=False,
)

# Create a layout for the prompt history and chat interface using the styles parameter
layout = pn.Row(
     chat_interface,
     prompt_history_layout, 
     css_classes=['box-shadow'],
     styles={'background': 'white'},
     margin=(10, 10, 10, 10)
 )


# Mark the layout as servable
layout.servable()
