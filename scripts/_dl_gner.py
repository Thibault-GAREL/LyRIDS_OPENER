"""Pre-download GNER T5-base (default HF cache on C:)."""
from huggingface_hub import snapshot_download
print("Downloading dyyyyyyyy/GNER-T5-base...")
p = snapshot_download("dyyyyyyyy/GNER-T5-base")
print(f"OK: {p}")
