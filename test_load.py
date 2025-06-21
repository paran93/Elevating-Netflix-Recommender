import json
import os

print("Current directory:", os.getcwd())
print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.json')])

try:
    print("Attempting to load netflix_content.json...")
    with open('netflix_content.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("✅ JSON loaded successfully!")
    print(f"Keys in data: {list(data.keys())}")
    if 'shows' in data:
        print(f"Number of shows: {len(data['shows'])}")
        if data['shows']:
            print(f"First show title: {data['shows'][0].get('title', 'No title')}")
            print(f"First show keys: {list(data['shows'][0].keys())}")
    else:
        print("❌ No 'shows' key found in JSON")
        print("Available keys:", list(data.keys()))
except FileNotFoundError:
    print("❌ File 'netflix_content.json' not found")
except json.JSONDecodeError as e:
    print(f"❌ JSON decode error: {e}")
    print("This usually means the JSON file is corrupted or has invalid format")
except Exception as e:
    print(f"❌ Other error: {e}")