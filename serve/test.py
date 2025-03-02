import os
import time
import requests
import subprocess

# Paths
MODEL_STORE = "./"  # Directory containing your .mar files
MODEL_NAME = "gpt2-medium"  # Name of your GPT-2 model
MODEL_MAR = f"{MODEL_NAME}.mar"  # .mar file for your GPT-2 model
TORCHSERVE_PORT = 8081  # Port for TorchServe
CONFIG = "config.properties"


# Step 1: Start TorchServe
def start_torchserve():
    print("Starting TorchServe...")
    command = [
        "torchserve",
        "--start",
        "--model-store",
        MODEL_STORE,
        "--models",
        MODEL_MAR,
        "--disable-token-auth",
        # "--ts-config", CONFIG,
        "--ncs",  # Disable snapshot feature
    ]
    subprocess.run(command, check=True)
    time.sleep(20)  # Wait for TorchServe to start


# Step 2: Register the Model
def register_model():
    print(f"Registering model {MODEL_NAME}...")
    url = f"http://localhost:{TORCHSERVE_PORT}/models"
    files = {
        "model_name": (None, MODEL_NAME),
        "url": (None, MODEL_MAR),
        "initial_workers": (None, "1"),
        "synchronous": (None, "true"),
    }
    response = requests.post(url, files=files)
    print(f"Model registration response: {response.status_code}, {response.text}")


# Step 3: Send a Prediction Request
def send_prediction_request():
    print("Sending prediction request...")
    url = f"http://localhost:{TORCHSERVE_PORT}/predictions/{MODEL_NAME}"

    # Define the text prompt (Pass as a string, not a dictionary)
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nConvert the active sentence to passive: 'The chef cooks the meal every day.'"

    # Send the JSON payload
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        url, data=prompt, headers=headers
    )  # Use `data=` instead of `json=`

    print(f"Prediction response: {response.status_code}, {response.text}")


# Step 4: Stop TorchServe
def stop_torchserve():
    print("Stopping TorchServe...")
    subprocess.run(["torchserve", "--stop"], check=True)


# Main Execution
if __name__ == "__main__":
    try:
        # Start TorchServe
        start_torchserve()

        # Register the model
        register_model()

        # Send a prediction request
        send_prediction_request()

    finally:
        # Stop TorchServe
        stop_torchserve()
