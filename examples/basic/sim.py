import json

def simulate():
    # Hard-coded simulation data
    data = [{"feature": i, "label": i % 2} for i in range(100)]
    output_file = "simulation_data.json"
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"Simulation data saved to {output_file}")

if __name__ == "__main__":
    simulate()

