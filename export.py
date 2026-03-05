import torch
import torch.onnx
from SWH.model import SolarEfficiencyANNFWH

# Import your model class here
# from your_module import SolarModel 

# 1. Initialize the architecture
model = SolarEfficiencyANNFWH() # Replace with your actual class name

# 2. Load your saved weights
save_path = "experiments/models/solar_model.pth"
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

# 3. Set to evaluation mode (crucial for Dropout/BatchNorm layers)
model.eval()

# 4. Create dummy input data 
# Match the shape your model expects: (Batch_Size, Channels, Height, Width)
# Example: 1 image, 3 channels, 224x224 pixels
# 28 * 8 = 224. This would match the input side of the error.

dummy_input = torch.randn(1, 6)
# 5. Export the model
onnx_path = "experiments/models/solar_model.onnx"
torch.onnx.export(
    model,                  # The model being exported
    dummy_input,            # A tuple or tensor of sample inputs
    onnx_path,              # Where to save the file
    export_params=True,      # Store the trained parameter weights inside the file
    opset_version=12,       # Recommended version for broad compatibility
    do_constant_folding=True, # Optimization to simplify the graph
    input_names=['input'],   # Label for the input node
    output_names=['output'], # Label for the output node
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allow variable batch sizes
)

print(f"Model successfully exported to {onnx_path}")
