"""Download GliNER with HF cache redirected to D: drive."""
import os
os.environ['HF_HOME'] = 'D:\\.hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:\\.hf_cache'
print(f"HF_HOME = {os.environ['HF_HOME']}")
from gliner import GLiNER
print("Loading GliNER L...")
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
print("Loaded OK")
