import torch
import logging
import tiktoken
from ts.torch_handler.base_handler import BaseHandler
from utils import GPTModel, generate, text_to_token_ids, token_ids_to_text

logger = logging.getLogger(__name__)


class GPT2Handler(BaseHandler):
    def __init__(self):
        super(GPT2Handler, self).__init__()
        self.model = None
        self.tokenizer = None
        self.context_length = 1024

    def initialize(self, context):
        # Load model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_path = f"{model_dir}/gpt2-medium355M-sft.pth"

        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": self.context_length,
            "drop_rate": 0.0,
            "qkv_bias": True,
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16,
        }

        self.model = GPTModel(BASE_CONFIG)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def preprocess(self, data):
        logger.debug(f"Received data: {data}")

        # TorchServe wraps input in a list
        if isinstance(data, list):
            data = data[0]

        # If input is a dictionary (e.g., {'body': bytearray(...)})
        if isinstance(data, dict) and "body" in data:
            data = data["body"]  # Extract bytearray

        # If data is in bytearray format, decode it to string
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")

        # Ensure input is a string
        if not isinstance(data, str):
            raise ValueError(
                "Invalid input format. Expected a plain text string.", data
            )

        token_ids = text_to_token_ids(data, self.tokenizer)
        return token_ids

    def inference(self, input_data):
        with torch.no_grad():
            token_ids = generate(
                model=self.model,
                idx=input_data,
                max_new_tokens=35,
                context_size=self.context_length,
                eos_id=50256,
            )
        return token_ids

    def postprocess(self, inference_output):
        response = token_ids_to_text(inference_output, self.tokenizer)
        return [response]
