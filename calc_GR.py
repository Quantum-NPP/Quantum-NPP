import os
import json
import time

# Function to apply the Greedy Algorithm (sorted)
def greedy_algorithm(weights):
    sorted_weights = sorted(weights, reverse=True)
    sum1 = 0
    sum2 = 0
    solution = [0] * len(weights)  # A solution array to indicate the partition of weights
    for i, weight in enumerate(sorted_weights):
        if sum1 <= sum2:
            sum1 += weight
            solution[i] = 0  # Assign to the first partition
        else:
            sum2 += weight
            solution[i] = 1  # Assign to the second partition
    E = abs(sum1 - sum2)
    return solution, E

base_path = "."

# Define the function to iterate through files, apply Greedy, and save results with time tracking
def process_files_with_greedy(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    problem = data.get("problem", [])
                    
                    # Apply Greedy Algorithm and track time
                    start_time = time.time()
                    greedy_solution, greedy_E = greedy_algorithm(problem)
                    greedy_time = time.time() - start_time  # Time in seconds
                    
                    # Add Greedy solution, energy, and time to the JSON
                    data["greedy"] = {
                        "run1": {
                            "solution": greedy_solution,
                            "E": greedy_E,
                            "time": greedy_time
                        }
                    }
                    
                    # Save the updated data back to the file
                    with open(file_path, "w") as json_output_file:
                        json.dump(data, json_output_file, indent=4)
                    
                    print(f"Processed {file_path} with Greedy")

# Call the function to process files
process_files_with_greedy(base_path)
