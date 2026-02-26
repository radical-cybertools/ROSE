# simulation.py
import json
import logging
import random

from datasets import load_dataset


def simulate(output_file, sample_size=100):
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading Rotten Tomatoes dataset...")

    raw_dataset = load_dataset("rotten_tomatoes")
    dataset = [
        {"text": text, "label": label}
        for text, label in zip(raw_dataset["train"]["text"], raw_dataset["train"]["label"])
    ]

    # Save the entire dataset to file
    with open("dataset.json", "w") as f:
        json.dump(dataset, f)

    # Sample random data
    samples = random.sample(dataset, sample_size)
    with open(output_file, "w") as f:
        json.dump(samples, f)

    logging.info(f"Selected {sample_size} samples.")


if __name__ == "__main__":
    simulate("samples.json")
