from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Iterator
import pickle
import random
import uuid
import datetime
import os
from collections import deque

@dataclass
class Experience:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: dict = None

class ExperienceBank:
    def __init__(self, max_size: Optional[int] = None, session_id: Optional[str] = None):
        self.max_size = max_size
        self._experiences = deque(maxlen=max_size) if max_size else deque()
        self._rng = random.Random()
        
        # Generate unique session ID if not provided
        if session_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            self.session_id = f"experience_bank_{timestamp}_{unique_id}"
        else:
            self.session_id = session_id

    def add(self, experience: Experience) -> None:
        self._experiences.append(experience)
    
    def add_batch(self, experiences: List[Experience]) -> None:
        self._experiences.extend(experiences)
    
    def sample(self, batch_size: int, replace: bool = True) -> List[Experience]:
        if not self._experiences:
            return []
        
        if not replace and batch_size > len(self._experiences):
            raise ValueError(f"Cannot sample {batch_size} experiences without replacement from bank of size {len(self._experiences)}")
        
        if replace:
            return self._rng.choices(list(self._experiences), k=batch_size)
        else:
            return self._rng.sample(list(self._experiences), k=batch_size)
        
    def merge(self, other: 'ExperienceBank') -> 'ExperienceBank':
        new_max_size = None
        if self.max_size is not None and other.max_size is not None:
            new_max_size = max(self.max_size, other.max_size)
        elif self.max_size is not None:
            new_max_size = self.max_size
        elif other.max_size is not None:
            new_max_size = other.max_size
        
        merged_bank = ExperienceBank(max_size=new_max_size)
        merged_bank.add_batch(list(self._experiences))
        merged_bank.add_batch(list(other._experiences))
        
        return merged_bank
    
    def merge_inplace(self, other: 'ExperienceBank') -> None:
        self.add_batch(list(other._experiences))
    
    def clear(self) -> None:
        self._experiences.clear()
    
    def get_recent(self, n: int) -> List[Experience]:
        return list(self._experiences)[-n:] if n <= len(self._experiences) else list(self._experiences)
    
    def save(self, work_dir: str = ".", bank_file: str = None) -> str:
        if bank_file:
            filepath = os.path.join(work_dir, bank_file)
        else:
            filepath = os.path.join(work_dir, f"{self.session_id}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(list(self._experiences), f)
        return filepath
    
    @classmethod
    def load(cls, filepath: str, max_size: Optional[int] = None) -> 'ExperienceBank':
        try:
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
            
            bank = cls(max_size=max_size)
            bank.add_batch(experiences)
            return bank
        except FileNotFoundError:
            return cls(max_size=max_size)
    
    def __len__(self) -> int:
        return len(self._experiences)
    
    def __iter__(self) -> Iterator[Experience]:
        return iter(self._experiences)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Experience, List[Experience]]:
        if isinstance(index, slice):
            return list(self._experiences)[index]
        return list(self._experiences)[index]
    
def create_experience(state, action, reward, next_state, done, info=None) -> Experience:
    return Experience(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        info=info or {}
    )