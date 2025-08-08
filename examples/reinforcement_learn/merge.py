import os
import sys
from rose.rl.experience import ExperienceBank

def merge_banks(work_dir="."):
    """Find and merge all experience banks in directory."""
    
    # Find all experience bank files
    bank_files = []
    for filename in os.listdir(work_dir):
        if filename.startswith("experience_bank_") and filename.endswith(".pkl"):
            bank_files.append(os.path.join(work_dir, filename))
    
    if not bank_files:
        print("No experience banks found!")
        return
    
    print(f"Found {len(bank_files)} experience banks")
    
    # Create merged bank and load all files
    merged = ExperienceBank()
    total = 0
    
    for bank_file in bank_files:
        try:
            bank = ExperienceBank.load(bank_file)
            merged.merge_inplace(bank)
            total += len(bank)
            print(f"  Merged {len(bank)} from {os.path.basename(bank_file)}")
        except:
            print(f"  Failed to load {bank_file}")

    for bank_file in bank_files:
        try:
            os.remove(bank_file)
        except Exception as e:
            print(f"  Failed to delete {bank_file}: {e}")
    
    # Save merged bank
    output_path = os.path.join(work_dir, "experience_bank.pkl")
    merged.save('.', output_path)
    
    print(f"Saved {len(merged)} total experiences to experience_bank.pkl")

if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    merge_banks(work_dir)