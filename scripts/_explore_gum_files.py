"""List all conllu files + check splits.md for official train/dev/test."""
import requests

# List all files in dep/
r = requests.get("https://api.github.com/repos/amir-zeldes/gum/contents/dep?ref=master")
files = [i['name'] for i in r.json() if isinstance(i, dict) and i.get('name', '').endswith('.conllu')]
gentle_files = [f for f in files if f.startswith('GENTLE_')]
gum_files = [f for f in files if f.startswith('GUM_')]
print(f"Total conllu files: {len(files)}  (GENTLE: {len(gentle_files)}, GUM: {len(gum_files)})")
print(f"\nGENTLE genres (from filenames):")
genres = sorted(set(f.split('_')[1] for f in gentle_files))
for g in genres:
    n = sum(1 for f in gentle_files if f.startswith(f'GENTLE_{g}_'))
    print(f"  {g:<20} {n} docs")

print(f"\nGUM genres:")
genres_gum = sorted(set(f.split('_')[1] for f in gum_files))
for g in genres_gum:
    n = sum(1 for f in gum_files if f.startswith(f'GUM_{g}_'))
    print(f"  {g:<20} {n} docs")

# Try to get splits.md
print("\n--- splits.md (first 80 lines) ---")
r2 = requests.get("https://raw.githubusercontent.com/amir-zeldes/gum/master/splits.md")
for line in r2.text.splitlines()[:80]:
    print(line)
