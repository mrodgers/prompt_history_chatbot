#!/bin/bash


# Checking Python version (assuming Python 3.x is needed)
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=8  # Update this as needed

echo "Verifying python version before executing chatbot..."

# Function to compare Python versions
version_greater_equal() {
    [ "$1" = "$(echo -e "$1\n$2" | sort -V | tail -n1)" ]
}

# Getting the current Python version
CURRENT_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
CURRENT_PYTHON_MAJOR=$(echo $CURRENT_PYTHON_VERSION | cut -d. -f1)
CURRENT_PYTHON_MINOR=$(echo $CURRENT_PYTHON_VERSION | cut -d. -f2)

# Checking if the current Python version is sufficient
if ! version_greater_equal "$CURRENT_PYTHON_MAJOR.$CURRENT_PYTHON_MINOR" "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR"; then
    echo "Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR or higher is required."
    echo "You have Python $CURRENT_PYTHON_VERSION."
    exit 1
fi

echo "Correct Python version detected: Python $CURRENT_PYTHON_VERSION"

# Navigating to the script directory if stored in other than CWD
#cd /path/to/multi-llm-integration

# Activating virtual environment if you have one
# Uncomment the line below if you have a virtual environment
# source venv/bin/activate

# Installing required dependencies
echo "Installing required Python packages from requirements.txt..."
pip install -r requirements.txt

# Running the Python script
echo "Starting the chatbot..."
panel serve chatbot.py --autoreload --show
