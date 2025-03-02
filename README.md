# LLM PROD

LLM PROD is a project focused on fine-tuning and deploying large language models (LLMs) for text generation tasks. The project includes scripts for training, inference, and serving the model via a FastAPI-based web application. It also supports distributed training, model conversion for ONNX and TensorRT, and fine-tuning using techniques like Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## Project Structure

```
LLM PROD/
├── data/
│   ├── instruction-data.json
│   ├── instruction-data-with-response.json
│   └── instruction-data-with-preference-1.json
├── example/
│   ├── DP.py
│   ├── dpo-llm.ipynb (DPO fine-tuning for RLHF)
│   ├── sft-gpt-lavita-medical-qa.ipynb (SFT fine-tuning on Lavita Medical dataset)
│   ├── train_dp.py
│   └── train_multinode.py
├── onnx/
│   ├── convert.py
│   └── utils.py
├── serve/
│   ├── config.properties
│   ├── handler.py
│   ├── test.py
│   └── utils.py
├── TensoRT/
│   └── tensort_convert.py
├── visualization/
├── requirements.txt
├── fastapi_llm_app.py
├── gpt_download.py
├── inference.py
├── train_dp.py
├── train.py
├── utils.py
└── Dockerfile
```

## Features

- **Fine-tuning**:
  - **Supervised Fine-Tuning (SFT)**: Fine-tune GPT-2 models on custom datasets (e.g., Lavita Medical dataset).
  - **Direct Preference Optimization (DPO)**: Further fine-tune models using RLHF (Reinforcement Learning with Human Feedback).
- **Distributed Training**: Support for multi-GPU and multi-node training using PyTorch's `DistributedDataParallel`.
- **Inference**: Generate text using the fine-tuned model.
- **Serving**: Serve the model via a FastAPI-based web application or TorchServe.
- **Model Conversion**: Convert models to ONNX and TensorRT for optimized inference.
- **Visualization**: Plot training and validation losses.

## Setup

### Prerequisites

- Python 3.9
- PyTorch
- FastAPI
- ONNX (optional)
- TensorRT (optional)
- TorchServe (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LLM PROD.git
   cd LLM PROD
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the GPT-2 model weights:
   ```bash
   python gpt_download.py
   ```

## Usage

### Training

#### Supervised Fine-Tuning (SFT)
To fine-tune the GPT-2 model on your dataset (e.g., Lavita Medical dataset), use the `example/sft-gpt-lavita-medical-qa.ipynb` notebook or the `train.py` script:

```bash
python train.py --mode single_gpu --total_epochs 10 --batch_size 8 --learning_rate 5e-5
```

#### Distributed Training 
Currently  it supports single, multi and single node gpu
script:

```bash
python train_dp.py --mode single_gpu --total_epochs 10 --batch_size 8 --learning_rate 5e-5
```

### Inference

To generate text using the fine-tuned model, use the `inference.py` script:

```bash
python inference.py
```

### Deployment

#### FastAPI
To serve the model via FastAPI, run the `fastapi_llm_app.py` script:

```bash
uvicorn fastapi_llm_app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can send a POST request to `/generate-text/` with a JSON payload containing the prompt:

```json
{
  "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
}
```

#### TorchServe (Serverless)
To serve the model using TorchServe, follow these steps:

1. Package the model:
   ```bash
   torch-model-archiver --model-name gpt2-medium --version 1.0 --handler serve/handler.py --export-path ./ --extra-files weights/gpt2-medium355M-sft.pth
   ```

2. Start TorchServe:
   ```bash
   torchserve --start --model-store ./ --models gpt2-medium.mar
   ```

3. Test the model using the `serve/test.py` script:
   ```bash
   python serve/test.py
   ```

### Model Conversion

#### ONNX
To convert the model to ONNX format, use the `convert.py` script in the `onnx` directory:

```bash
python onnx/convert.py
```

#### TensorRT
To convert the model to TensorRT format, use the `tensort_convert.py` script in the `TensoRT` directory:

```bash
python TensoRT/tensort_convert.py
```

## Visualization

Training and validation losses are automatically plotted and saved in the `visualization` directory as `loss-plot.pdf`.

## Docker

You can also run the application in a Docker container:

1. Build the Docker image:
   ```bash
   docker build -t llm_prod .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 llm_prod
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

### Key Updates:
1. **Project Structure**: Added details about the `dpo-llm.ipynb` and `sft-gpt-lavita-medical-qa.ipynb` files.
2. **Features**: Highlighted SFT and DPO fine-tuning techniques.
3. **Usage**: Added instructions for serving the model using TorchServe.
4. **Training**: Included details about SFT and DPO fine-tuning.
5. **Serving**: Added FastAPI and TorchServe serving options.

This README provides a comprehensive guide for anyone looking to use or contribute to your project.
