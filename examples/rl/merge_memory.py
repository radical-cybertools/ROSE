import pickle
import glob
from collections import deque, namedtuple

# Configuration
MEMORY_FILES_PATTERN = '/home/andrew/HPC/ROSE/examples/rl/data/replay_memory_*.pkl'  # Adjust to match your naming scheme
OUTPUT_FILE = '/home/andrew/HPC/ROSE/examples/rl/data/replay_memory.pkl'
MAX_CAPACITY = int(1e5)  # Adjust as needed

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def load_memory(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def merge_memories(memory_files, max_capacity):
    merged = deque(maxlen=max_capacity)
    for file in memory_files:
        print(f"Loading: {file}")
        mem = load_memory(file)
        merged.extend(mem)
    return merged

if __name__ == "__main__":
    memory_files = sorted(glob.glob(MEMORY_FILES_PATTERN))
    if not memory_files:
        print("No memory files found.")
    else:
        merged_memory = merge_memories(memory_files, MAX_CAPACITY)
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(merged_memory, f)
        print(f"Merged {len(memory_files)} files into {OUTPUT_FILE} ({len(merged_memory)} total experiences).")
