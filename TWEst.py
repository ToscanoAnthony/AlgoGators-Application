import os

# Get current working directory
print("Current working directory:", os.getcwd())
print("\nFiles in this directory:")
for file in os.listdir():
    if file.endswith(('.csv', '.png')):
        print(f"  ✓ {file}")