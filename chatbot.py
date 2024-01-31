# Importing necessary libraries
import panel as pn  # For creating web app UI
import datetime as dt  # To handle dates and times
from panel.widgets import DatetimeRangeInput  # Date range selector widget
from ctransformers import AutoModelForCausalLM  # Import for local language models
import openai  # OpenAI's API client
from anthropic import (
    Anthropic,
    HUMAN_PROMPT,
    AI_PROMPT,
)  # Anthropic API client and constants
import vertexai  # Vertex AI client for Google's AI models
from vertexai.preview.generative_models import (
    GenerativeModel,
    Part,
)  # Specific modules for Vertex AI
import time  # For time tracking (e.g., response latency)
import csv  # For CSV file operations
import os  # For interacting with the operating system (e.g., environment variables)
from dotenv import load_dotenv  # To load environment variables from a .env file
import requests  # For making HTTP requests
import json  # For handling JSON data

# Load environment variables from .env file for API keys and configurations
load_dotenv()

# Accessing the API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initializing Panel UI with Material Design template
pn.extension(template="material")
pn.state.template.param.update(title="Prompt History Chatbot Demo")

# Retrieving Google Vertex AI project and location IDs from environment variables
google_project_id = os.getenv("GOOGLE_PROJECT_ID")
google_location_id = os.getenv("GOOGLE_LOCATION_ID")

# Retrieving PromptMule's API key from environment variables
promptmule_api_key = os.getenv("PROMPTMULE_API_KEY")

# Specifying the filename where responses along with latencies will be saved
response_filename = "multi_llm_responses.csv"

# Displaying retrieved API keys and setup information
print("OpenAI API Key:", openai_api_key)
print("Anthropic API Key:", anthropic_api_key)
print("Google Setup: ", google_location_id, google_project_id)
print("PromptMule API Key: ", promptmule_api_key)

# Defining model arguments for local language models
MODEL_ARGUMENTS = {
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


# Function to generate text using Google Vertex AI's generative model
def generate_text_with_vertex_ai(project_id: str, location: str, query: str) -> str:
    # Handling the absence of project ID or location ID
    if not project_id:
        print("Skipping Vertex AI due to missing Google Project ID.")
        return "Skipping Vertex AI due to missing Google Project ID."
    if not location:
        print("Skipping Vertex AI due to missing Google Project Location.")
        return "Skipping Vertex AI due to missing Google Project Location."

    # Initializing Vertex AI with the given project and location IDs
    vertexai.init(project=project_id, location=location)
    multimodal_model = GenerativeModel("gemini-pro-vision")

    print("Sending prompt to: Vertex AI")
    response = multimodal_model.generate_content([Part.from_text(query)])
    return response.text


def generate_text_with_promptmule(promptmule_api_key, contents):
    url = "https://api.promptmule.com/prompt"
    headers = {"x-api-key": promptmule_api_key, "Content-Type": "application/json"}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": contents}],
        "max_tokens": 512,
        "api": "openai",
        "semantic": "0.99",
        "sem_num": "1"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        response_json = response.json()
        promptmule_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        promptmule_cache = response_json.get("choices", [{}])[0].get("is_cached")
        return promptmule_cache, promptmule_response
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
        return None, f"HTTP Error: {http_err}"
    except Exception as e:
        print(f"Error: {e}")
        return None, f"Error: {e}"


# Function to generate text using Anthropic's Claude 2.1
def generate_text_with_anthropic(api_key, prompt):
    if not api_key:
        print("Skipping Anthropic due to missing API key.")
        return "Anthropic API key not provided."

    print("Sending prompt to: Anthropic")
    anthropic = Anthropic(api_key=api_key)

    try:
        response = anthropic.completions.create(
            model="claude-2.1",
            max_tokens_to_sample=512,
            temperature=0,
            prompt=f"{HUMAN_PROMPT} {prompt}\n{AI_PROMPT}",
        )
        return response
    except Exception as e:
        return f"An error occurred: {e}"


# Function to generate text using OpenAI's GPT-3
def generate_text_with_openai(api_key, prompt):
    if not api_key:
        print("Skipping OpenAI due to missing API key.")
        return "OpenAI API key not provided, get one at https://openai.com"

    openai.api_key = api_key
    print("Sending prompt to: OpenAI")

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


# Function to write the gathered data to a CSV file
def write_to_csv(prompt, responses, latencies, filename):
    # Check if the file already exists
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        # Write headers if the file is being created
        if not file_exists:
            headers = (
                ["Prompt"]
                + [f"Response from LLM {i+1}" for i in range(len(responses))]
                + [f"Latency for LLM {i+1}" for i in range(len(latencies))]
            )
            writer.writerow(headers)
        # Write the data
        writer.writerow([prompt] + responses + latencies)


# Callback function to handle date range input for prompt history
def create_on_date_range_change(
    token, api_key, prompt_display_pane, quantity_select, prompt_count_pane
):
    def on_date_range_change(event):
        # Extracting new date range from the event
        if isinstance(event, dict):
            start_date, end_date = event["new"]
        else:
            start_date, end_date = event.new

        print("Selected Date Range:", start_date, "to", end_date)

        # Formatting dates for API call
        formatted_start_date = start_date.strftime("%Y-%m-%d")
        formatted_end_date = end_date.strftime("%Y-%m-%d")

        num_prompts = quantity_select.value
        recent_prompts = get_date_based_prompt_history(
            token, api_key, formatted_start_date, formatted_end_date, num_prompts
        )

        if not recent_prompts:
            print("Retrieved Prompt History with Zero Prompt Results")
            recent_prompts = [
                {
                    "request-time": formatted_start_date,
                    "prompt": "Range Empty for Prompt History",
                }
            ]
        else:
            print("Retrieved Prompt History.")

        # Updating prompt display pane with the retrieved prompts
        prompt_display = "\n\n".join(
            [f"{p['request-time']}: {p['prompt']}" for p in recent_prompts]
        )
        prompt_display_pane.object = prompt_display
        actual_count = len(recent_prompts)
        prompt_count_pane.object = f"Ask: {num_prompts} Retrieved: {actual_count}"

    return on_date_range_change


# Function to fetch prompt history based on date range
def date_based_prompt_history(token, api_key, start_date, end_date):
    url = (
        f"https://api.promptmule.com/prompt?start-date={start_date}&end-date={end_date}"
    )
    headers = {"x-api-key": api_key, "Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, verify=True)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except Exception as err:
        return f"Other error occurred: {err}"


# Function to perform login to PromptMule
def login_promptmule(username, password):
    url = "https://api.promptmule.com/login"
    headers = {"Content-Type": "application/json"}
    data = {"username": username, "password": password}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("success"):
            return response_data.get("token")
        else:
            return response_data.get("message")
    except requests.RequestException as e:
        return f"Error during login: {e}"


# Function to fetch date-based prompt history
def get_date_based_prompt_history(
    token, api_key, formatted_start_date, formatted_end_date, num_prompts=10
):
    # Calculate date range and fetch prompts
    end_date = formatted_end_date
    start_date = formatted_start_date

    try:
        response = date_based_prompt_history(
            token, api_key, str(start_date), str(end_date)
        )
        if not response["prompts"]:
            print("Prompt history is empty.")
            return []
        else:
            print("Prompts retrieved.")
            if isinstance(response, dict) and "prompts" in response:
                return response["prompts"][:num_prompts]
            else:
                print("Unexpected response format or error:", response)
                return []
    except Exception as e:
        print(f"An error occurred while fetching prompt history: {e}")
        return []


# Function to retrieve similar prompts based on semantic similarity
def get_similar_prompts(user_prompt, api_key, model, sem_score_max, num_prompts=10):
    try:
        response = similar_prompts(
            user_prompt, api_key, model, sem_score_max, num_prompts
        )
        print("in get_similar_prompts(), response:", response)
        if "prompt" in response and response["prompt"]:
            print("Similar prompts retrieved.")
            similar_prompts_list = response
            return similar_prompts_list
        else:
            print("No similar prompts found or empty response.")
            return []
    except Exception as e:
        print(f"An error occurred while fetching similar prompts: {e}")
        return []


# Function to initiate a search for similar prompts based on user input
def search_similar_prompts(event):
    user_prompt = prompt_search_box.value
    if user_prompt:
        model = "gpt-3.5-turbo"
        sem_score = str(sem_score_slider.value)
        sem_num = "10"

        similar_prompts_response = get_similar_prompts(
            user_prompt, promptmule_api_key, model, sem_score, sem_num
        )
        print("in search_similar_prompts(), response:", similar_prompts_response)

        if "choices" in similar_prompts_response:
            display_text = "\n\n".join(
                [
                    f"Response: {choice['message']['content']} Similarity: {choice.get('score', 'N/A')}"
                    for choice in similar_prompts_response["choices"]
                ]
            )
            prompt_display_pane.object = display_text
            prompt_count_pane.object = (
                f"Retrieved {len(similar_prompts_response['choices'])} similar prompts."
            )
        else:
            prompt_display_pane.object = "No similar prompts found."
            prompt_count_pane.object = f"Retrieved: 0"
    else:
        prompt_display_pane.object = "Enter a prompt for similarity search."
        prompt_count_pane.object = ""


# Function to make API call to PromptMule for fetching similar prompts
def similar_prompts(user_prompt, api_key, model, sem_score, sem_num):
    url = "https://api.promptmule.com/prompt"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 500,
        "temperature": 0,
        "api": "openai",
        "semantic": sem_score,
        "sem_num": sem_num,
    }

    print("payload: ", payload)
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print("in similar_prompts(), response:", response)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        return f"HTTP Error: {err}"
    except Exception as e:
        return f"Error: {e}"


# Callback function for handling chat interactions
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    responses = []
    latencies = []
    model_names = []

    # Define prompts for local LLMs
    alpaca_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {contents}

    ### Response:"""

    llama_prompt = f"""
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant.
    <</SYS>>
    {contents} [/INST]</s>"""

    mistral_prompt = (
        f"""<s>[INST] Please respond to the Question : {contents} [/INST]"""
    )

    # Interacting with each API and local LLMs
    # OpenAI response
    print("OpenAI Key: ", openai_api_key)
    start_time = time.time()
    openai_response = generate_text_with_openai(openai_api_key, contents)
    openai_latency = time.time() - start_time

    model_names.append("OpenAI")
    if openai_api_key:
        responses.append(openai_response)
        latencies.append(openai_latency)
        instance.stream(
            openai_response + "\nlatency: " + f"{openai_latency:.3f}",
            user="OpenAI",
            message=None,
        )
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
        instance.stream(
            anthropic_response.completion + "\nlatency: " + f"{anthropic_latency:.3f}",
            user="Anthropic",
            message=None,
        )
    else:
        responses.append("")
        latencies.append(0)
        instance.stream("Anthropic API Key not found.", user="System", message=None)

    # Google Vertex response
    model_names.append("Google Vertex AI")
    if google_project_id:
        if google_location_id:
            start_time = time.time()
            vertex_ai_response = generate_text_with_vertex_ai(
                google_project_id, google_location_id, contents
            )
            vertex_ai_latency = time.time() - start_time
            responses.append(vertex_ai_response)
            latencies.append(vertex_ai_latency)
            instance.stream(
                vertex_ai_response + "\nlatency: " + f"{vertex_ai_latency:.3f}",
                user="Google Vertex AI",
                message=None,
            )
        else:
            responses.append("")
            latencies.append(0)
            instance.stream(
                "Missing Vertex AI: Google Project Location",
                user="System",
                message=None,
            )
    else:
        responses.append("")
        latencies.append(0)
        instance.stream(
            "Missing Vertex AI: Google Project ID", user="System", message=None
        )

    # PromptMule response
    model_names.append("PromptMule")
    if promptmule_api_key:
        start_time = time.time()
        promptmule_cache, promptmule_response = generate_text_with_promptmule(
            promptmule_api_key, contents
        )
        promptmule_latency = time.time() - start_time
        responses.append(promptmule_response)
        latencies.append(promptmule_latency)
        instance.stream(
            promptmule_response + "\nlatency: " + f"{promptmule_latency:.3f}" + "\ncached: " + str(promptmule_cache),
            user="PromptMule"
         )
    else:
        responses.append("")
        latencies.append(0)
        instance.stream(
            "PromptMule API Key not set. Get one at https://app.promptmule.com",
            user="System",
            message=None,
        )

    # Handling responses from local LLMs
    for model in MODEL_ARGUMENTS:
        try:
            if model not in pn.state.cache:
                print(f"Loading model: {model}")
                pn.state.cache[model] = AutoModelForCausalLM.from_pretrained(
                    *MODEL_ARGUMENTS[model]["args"],
                    **MODEL_ARGUMENTS[model]["kwargs"],
                    gpu_layers=30,
                )

            llm = pn.state.cache[model]
            if model == "samantha":
                prompt = alpaca_prompt
            elif model == "llama":
                prompt = llama_prompt
            elif model == "mistral":
                prompt = mistral_prompt
            else:
                prompt = "Default prompt text"

            print("Sending prompt to: ", model)
            start_time = time.time()
            response = llm(prompt, max_new_tokens=512, stream=False)
            model_latency = time.time() - start_time
            instance.stream(
                response.strip() + "\nlatency: " + f"{model_latency:.3f}",
                user=model.title(),
            )
            model_names.append(model)
            responses.append(response.strip())
            latencies.append(model_latency)

        except Exception as e:
            print(f"Error processing model {model}: {e}")

    # Summarizing latencies
    max_name_length = max(len(name) for name in model_names)
    table_rows = "\n".join(
        [
            f"{model:<{max_name_length}} was {latency:.3f}sec"
            for model, latency in zip(model_names, latencies)
        ]
    )
    instance.stream(table_rows, user="LLM Latency Report")

    # Writing responses and latencies to CSV
    write_to_csv(contents, responses, latencies, response_filename)


# Custom CSS to enhance UI and make prompt history pane scrollable
pn.config.raw_css.append(
    """
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
"""
)
# Instantiate widgets
quantity_select = pn.widgets.Select(
    name="Prompts to Retrieve", options=[1, 10, 50, 100, 500, 1000], value=10, width=125
)
datetime_range_picker = pn.widgets.DatetimeRangePicker(
    name="Prompt Creation Date (GMT)",
    value=(dt.datetime.now() - dt.timedelta(days=7), dt.datetime.now()),
    margin=(10, 10, 10, 10),
    css_classes=["box-shadow", "datetime-picker-font"],
    styles={"background": "white"},
)
prompt_display_pane = pn.pane.Markdown(
    name="Recent Prompts",
    styles={"background": "white"},
    margin=(10, 10, 10, 10),
    css_classes=["box-shadow", "scrollable-pane"],
)
prompt_count_pane = pn.pane.Markdown("", name="Retrieved Count")
prompt_search_box = pn.widgets.TextInput(
    name="Prompt Explorer",
    placeholder="Enter your prompt here to discover similar responses...",
)
sem_score_slider = pn.widgets.FloatSlider(
    name="Similarity Percentage", start=0.1, end=1.0, step=0.01, value=0.95
)
search_button = pn.widgets.Button(name="Search", button_type="primary")
search_button.on_click(search_similar_prompts)

# Create the on_date_range_change callback with the obtained token
token = login_promptmule(
    os.getenv("PROMPTMULE_USERNAME"), os.getenv("PROMPTMULE_PASSWORD")
)
on_date_range_change = create_on_date_range_change(
    token, promptmule_api_key, prompt_display_pane, quantity_select, prompt_count_pane
)
datetime_range_picker.param.watch(on_date_range_change, "value")
on_date_range_change(
    {"new": (dt.datetime.now() - dt.timedelta(days=7), dt.datetime.now())}
)

# Initializing widgets for prompt history and search interface
default_start_date = dt.datetime.now() - dt.timedelta(days=7)  # 7 days before today
default_end_date = dt.datetime.now()  # Today
quantity_select = pn.widgets.Select(
    name="Prompts to Retrieve", options=[1, 10, 50, 100, 500, 1000], value=10, width=125
)
datetime_range_picker = pn.widgets.DatetimeRangePicker(
    name="Prompt Creation Date (GMT)",
    value=(default_start_date, default_end_date),
    margin=(10, 10, 10, 10),
    css_classes=["box-shadow", "datetime-picker-font"],
    styles={"background": "white"},
)
datetime_range_picker.param.watch(on_date_range_change, "value")
on_date_range_change({"new": (default_start_date, default_end_date)})
datetime_range_picker.css_classes = ["material-select"]
quantity_select.css_classes = ["material-select"]
prompt_display_pane.css_classes = ["material-pane", "scrollable-prompt-pane"]
prompt_count_pane.css_classes = ["material-label"]
prompt_search_box = pn.widgets.TextInput(
    name="Prompt Explorer",
    placeholder="Enter your prompt here to discover similar responses...",
)
search_button = pn.widgets.Button(name="Search", button_type="primary")
search_button.on_click(search_similar_prompts)


# Creating layout for prompt history and chat interface
chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send(
    "Send a message to get a response from each: PromptMule, Llama 2, Mistral (7B), Google, Anthropic and OpenAI!",
    user="System",
    respond=False,
)

prompt_history_layout = pn.Column(
    pn.pane.Markdown("##  Search", css_classes=['widget-title']),
    # Placing search box and button in the same row for better alignment
    pn.Row(prompt_search_box, search_button, css_classes=['material-row']),
    pn.Row(sem_score_slider, css_classes=['material-row']),  # Add the semantic score slider
    pn.Row(pn.pane.Markdown(css_classes=['material-label']), datetime_range_picker, css_classes=['material-row']),
    pn.Row(pn.pane.Markdown(css_classes=['material-label']), quantity_select, css_classes=['material-row']),
    prompt_display_pane,
    prompt_count_pane,
    css_classes=['material-pane', 'fixed-width-prompt-history'],  # Added class
    margin=(10, 10, 10, 10)
)


layout = pn.Row(
    chat_interface,
    prompt_history_layout,
    css_classes=["box-shadow"],
    styles={"background": "white"},
    margin=(10, 10, 10, 10),
)
layout.servable()  # Mark the layout as servable for web app deployment
