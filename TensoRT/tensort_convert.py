import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes the CUDA context

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
with open("gpt2_medium.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# Get input and output shapes
input_shape = engine.get_tensor_shape(
    "input_ids"
)  # Ensure this matches your ONNX model
output_shape = engine.get_tensor_shape("logits")

# Allocate memory on the GPU
d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.dtype(np.int32).itemsize))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float32).itemsize))

# Create CUDA stream
stream = cuda.Stream()


# Run inference
def infer(input_data):
    input_data = input_data.astype(np.int32)
    output_data = np.empty(output_shape, dtype=np.float32)

    # Transfer input data to GPU
    cuda.memcpy_htod_async(d_input, input_data, stream)

    # Execute inference
    context.execute_v2(bindings=[int(d_input), int(d_output)])

    # Transfer output data back to CPU
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    return output_data


# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Example inference
input_data = np.random.randint(0, 50257, size=input_shape, dtype=np.int32)
output_logits = infer(input_data)

# Apply softmax to get probabilities
output_probs = softmax(output_logits)

# Get the token with the highest probability
output_tokens = np.argmax(output_probs, axis=-1)

print("Output logits shape:", output_logits.shape)
print("Output tokens:", output_tokens)
print("✅ TensorRT inference successful!")
