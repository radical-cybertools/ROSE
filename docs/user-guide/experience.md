# ROSE Experience and Experience Banks

ROSE provides a comprehensive experience management system for Reinforcement Learning workflows through the `Experience` dataclass and `ExperienceBank` class.

## Experience Data Structure

The `Experience` class captures a single environment interaction containing:
- **State** - Current environment observation
- **Action** - Action taken by the agent
- **Reward** - Reward received from environment
- **Next State** - Resulting environment observation
- **Done** - Boolean indicating episode termination
- **Info** - Optional metadata dictionary

Use the helper function to create experiences during environment interactions:
```python
from rose.rl.experience import Experience, create_experience

experience = create_experience(state, action, reward, next_state, done, info={"step": 42})
```
## Experience Bank

Experience Banks provide persistent storage and efficient sampling for RL experiences:

```python
from rose.rl.experience import ExperienceBank

# Create a bank with size limit
bank = ExperienceBank(max_size=10000)

# Add experiences
bank.add(experience)
# Add collection of experiences
bank.add_batch([experience1, experience2, experience3])

# Sample for training
batch = bank.sample(batch_size=32, replace=True)
```

ROSE assigns experience banks a unique session ID automatically, or session IDs can be assigned manually. The session ID is used to identify the bank and can be used to save/load the bank to/from disk:

```python
# Custom session ID
bank = ExperienceBank(session_id="rose_session")

# Save with auto-generated filename
filepath = bank.save() # Output: experience_bank_{date}_{session_id}.pkl

# Save with custom filename
filepath = bank.save(work_dir="./data", bank_file="experiences.pkl")

# Load from file
loaded_bank = ExperienceBank.load("./data/experiences.pkl")
```

Experience banks can be merged, allowing for efficient combination of experiences from multiple sources:

```python
# Create new merged bank
merged_bank = bank1.merge(bank2)

# Merge bank2 into bank1 (modifies bank1)
bank1.merge_inplace(bank2)

# Find all bank files in directory
bank_files = ExperienceBank.list_saved_banks("./data")
```
