import os

file_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

if os.path.exists(file_path):
    print(f"✅ File exists: {file_path}")
else:
    print(f"❌ File does not exist: {file_path}")
