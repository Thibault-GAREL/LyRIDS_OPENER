"""Explore GUM/GENTLE conllu format to find NER annotations."""
import requests

# Try a GENTLE file
url = "https://raw.githubusercontent.com/amir-zeldes/gum/master/dep/GENTLE_dictionary_next.conllu"
r = requests.get(url)
lines = r.text.splitlines()
print(f"GENTLE_dictionary_next: {len(lines)} lines\n")
# Show first 60 non-comment lines + a few comments
shown = 0
for i, l in enumerate(lines):
    if l.startswith('#') and shown < 8:
        print(f"{i:3} | {l}")
    elif not l.startswith('#') and shown < 30:
        print(f"{i:3} | {l[:200]}")
        shown += 1
    if shown >= 30:
        break

print("\n--- Columns (CoNLL-U has 10 cols: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC) ---")
print("NER tags in CoNLL-U are usually in MISC column under key 'Entity=' or in a custom column.")
