import json
import os
import sys

def check_mock():
    print("--- Environment Diagnosis ---")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Executable: {sys.executable}")
    
    # Use the script's directory as the base to find files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script Directory: {script_dir}")
    
    # Constructed path to the mock file - Now in the same directory
    mock_file_path = os.path.join(script_dir, "mock_stt_data.json")
    print(f"Target Mock File: {mock_file_path}")
    
    if not os.path.exists(mock_file_path):
        print(f"\n[!] Error: File not found at {mock_file_path}")
        return

    size = os.path.getsize(mock_file_path)
    print(f"File size: {size} bytes")
    
    if size == 0:
        print("\n[!] Error: The file is empty (0 bytes). This explains the JSONDecodeError.")
        return

    print(f"\n--- Loading Attempt ---")
    try:
        with open(mock_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Content read successfully ({len(content)} characters).")
            
            if not content.strip():
                print("[!] Error: File content is only whitespace or empty.")
                return
            
            # Show first few characters to check for BOM or weirdness
            print(f"Start of content: {repr(content[:50])}")
            
            data = json.loads(content)
            print("\n[SUCCESS] JSON loaded successfully!")
            print(f"Available keys: {list(data.keys())}")
            
            if "room_1" in data:
                print("Found 'room_1' in mock data.")
            else:
                print("WARNING: 'room_1' not found in mock data.")
                
    except json.JSONDecodeError as e:
        print(f"\n[!] JSON Decode Error: {e}")
        print(f"Error occurred at line {e.lineno}, column {e.colno} (char {e.pos})")
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")

if __name__ == "__main__":
    check_mock()
