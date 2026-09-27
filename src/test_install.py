import transformers
import torch
import pandas as pd
import numpy as np
import fairlearn

print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"fairlearn: {fairlearn.__version__}")
print("All dependencies installed successfully!")