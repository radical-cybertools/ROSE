import datetime
import os
import pickle
import random
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class Experience:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: dict = None


class ExperienceBank:
    """A memory bank for storing and sampling Experience objects.

    The ExperienceBank supports adding individual or batches of
    experiences, sampling with or without replacement, merging with
    other banks, saving/loading to disk, and retrieving recent
    experiences. It uses a deque for efficient memory management
    and can be bounded by a maximum size.

    Attributes:
        max_size (Optional[int]): Maximum number of experiences to store.
        If None, unlimited.
        session_id (str): Unique identifier for the bank session.

    Methods:
        add(experience): Add a single Experience to the bank.
        add_batch(experiences): Add multiple Experience objects.
        sample(batch_size, replace): Randomly sample experiences.
        merge(other): Return a new ExperienceBank merged with another.
        merge_inplace(other): Merge another bank into this one in-place.
        clear(): Remove all experiences.
        get_recent(n): Get the n most recent experiences.
        save(work_dir, bank_file): Save the bank to disk.
        load(filepath, max_size): Load a bank from disk.
    """

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

    def add_batch(self, experiences: list[Experience]) -> None:
        self._experiences.extend(experiences)

    def sample(self, batch_size: int, replace: bool = True) -> list[Experience]:
        if not self._experiences:
            return []

        if not replace and batch_size > len(self._experiences):
            raise ValueError(
                f"Cannot sample {batch_size} experiences "
                f"without replacement from bank of size {len(self._experiences)}"
            )

        if replace:
            return self._rng.choices(list(self._experiences), k=batch_size)
        else:
            return self._rng.sample(list(self._experiences), k=batch_size)

    def merge(self, other: "ExperienceBank") -> "ExperienceBank":
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

    def merge_inplace(self, other: "ExperienceBank") -> None:
        self.add_batch(list(other._experiences))

    def clear(self) -> None:
        self._experiences.clear()

    def get_recent(self, n: int) -> list[Experience]:
        return (
            list(self._experiences)[-n:] if n <= len(self._experiences) else list(self._experiences)
        )

    def save(self, work_dir: str = ".", bank_file: str = None) -> str:
        if bank_file:
            filepath = os.path.join(work_dir, bank_file)
        else:
            filepath = os.path.join(work_dir, f"{self.session_id}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(list(self._experiences), f)
        return filepath

    @classmethod
    def load(cls, filepath: str, max_size: Optional[int] = None) -> "ExperienceBank":
        """Load an ExperienceBank from a pickle file.

        Args:
            filepath (str): Path to the pickle file containing
            a list of Experience objects.
            max_size (Optional[int]): Maximum size of the loaded ExperienceBank.
                If specified, the bank will be initialized with this max size.
                Defaults to None.
        Returns:
            ExperienceBank: An instance of ExperienceBank populated with experiences
            loaded from the file.
            If the file does not exist, returns an empty ExperienceBank with the
            specified max_size.
        Raises:
            Any exception raised by pickle.load except FileNotFoundError, which
            is handled internally.
        """
        try:
            with open(filepath, "rb") as f:
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

    def __getitem__(self, index: Union[int, slice]) -> Union[Experience, list[Experience]]:
        if isinstance(index, slice):
            return list(self._experiences)[index]
        return list(self._experiences)[index]


def create_experience(state, action, reward, next_state, done, info=None) -> Experience:
    """
    Create an Experience object with the given parameters.
    Args:
        state (Any): The current state.
        action (Any): The action taken.
        reward (float): The reward received after taking the action.
        next_state (Any): The resulting state after the action.
        done (bool): Whether the episode has ended.
        info (dict, optional): Additional information about the experience.
        Defaults to None.

    Returns:
        Experience: An instance of the Experience dataclass containing
        the provided data.
    """
    return Experience(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        info=info or {},
    )
