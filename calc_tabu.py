import os
import json
import time
import dimod
from tabu import TabuSampler
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

# Function to generate BQM for Number Partitioning Problem (NPP)
def generate_NPP_bqm(weights):
    c = sum(weights)
    Q = {}
    
    for i in range(len(weights)):
        for j in range(len(weights)):
            Q[(i, j)] = 4 * weights[i] * weights[j]
    
    for i in range(len(weights)):
        Q[(i, i)] = 4 * weights[i] * (weights[i] - c)
    
    return dimod.BinaryQuadraticModel.from_qubo(Q=Q, offset=c**2)

def calculate_diff(solution, weights):
    solution = {s: (2*state-1) for s, state in solution.items()}
    spins = [solution[i] for i in range(len(weights))]
    return abs(sum(w if s == +1 else -w for w, s in zip(weights, spins)))

# Tabu search function to be run multiple times with execution time tracking
def run_tabu_once(weights):
    bqm = generate_NPP_bqm(weights)
    
    # Set up the Tabu Sampler
    sampler = TabuSampler()

    # Track start time
    start_time = time.time()

    # Run the Tabu Search on the BQM
    sampleset = sampler.sample(bqm)

    # Track end time
    end_time = time.time()
    execution_time = end_time - start_time

    # Extract the best solution found and convert values to standard int for JSON serialization
    best_solution = {k: int(v) for k, v in sampleset.first.sample.items()}  # Convert to standard Python int
    best_energy = sampleset.first.energy ** (0.5)

    if abs(best_energy - calculate_diff(best_solution, weights)) > 1e-6:
        print("WARNING: Energy does not match the calculated difference!", flush=True)
        print(f"Energy: {best_energy}, Calculated Difference: {calculate_diff(best_solution, weights)}", flush=True)

    # Return the result with execution time
    return {"solution": best_solution, "E": best_energy, "time": execution_time}

# Retry mechanism with exception handling
def retry_on_exception(func, *args, retries=5, wait_time=10, **kwargs):
    """Retries the function call if an exception occurs."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Exception occurred: {e}. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{retries})", flush=True)
                sleep(wait_time)
            else:
                print(f"Failed after {retries} retries.", flush=True)
                raise

# Function to process and run 5 parallel Tabu executions for each problem
def process_files_with_tabu(base_path):
    # Define the number of parallel runs and threads
    num_runs = 5
    max_threads = 3

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} with Tabu Search", flush=True)

                try:
                    # Load the JSON data
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        problem = data.get("problem", [])

                    # Parallel execution using ThreadPoolExecutor
                    results = {}
                    with ThreadPoolExecutor(max_workers=max_threads) as executor:
                        futures = [executor.submit(retry_on_exception, run_tabu_once, problem) for _ in range(num_runs)]
                        
                        # Collect results as futures complete
                        for i, future in enumerate(as_completed(futures), 1):
                            results[f"run{i}"] = future.result()

                    # Save results back to the JSON file
                    data["tabu"] = results

                    # Write updated data back to JSON
                    with open(file_path, "w") as json_output_file:
                        json.dump(data, json_output_file, indent=4)

                    print(f"Processed {file_path} with Tabu Search", flush=True)

                except json.JSONDecodeError as e:
                    print(f"Error processing {file_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error occurred: {e}")

# Set the path to the folder containing JSON problem files
base_path = "."

# Run the processing function
process_files_with_tabu(base_path)
