import os
import json
import time
import random
import dimod
from tabu import TabuSampler
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

# Helper to retry in case of exceptions
def retry_on_exception(func, *args, retries=20, wait_time=30, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Exception occurred: {e}. Retrying in {wait_time} seconds...")
                sleep(wait_time)
            else:
                raise

# Generate the BQM for Number Partitioning Problem
def generate_NPP_bqm(weights):
    c = sum(weights)
    Q = {}
    
    for i in range(len(weights)):
        for j in range(len(weights)):
            Q[(i, j)] = 4 * weights[i] * weights[j]
    
    for i in range(len(weights)):
        Q[(i, i)] = 4 * weights[i] * (weights[i] - c)
    
    return dimod.BinaryQuadraticModel.from_qubo(Q=Q, offset=c**2)

# Function to split the problem into subproblems and indices
def gen_sub_qubo_with_indices(weights, n_sub):
    indexed_weights = list(enumerate(weights))
    random.shuffle(indexed_weights)  # Shuffling with reproducibility

    indices, sub_problems = [], []
    per_sub = len(indexed_weights) // n_sub
    remainder = len(indexed_weights) % n_sub

    for i in range(n_sub):
        extra = 1 if i < remainder else 0
        start = i * per_sub + min(i, remainder)
        end = start + per_sub + extra
        sub_indices, sub_prob = zip(*indexed_weights[start:end])
        indices.append(list(sub_indices))
        sub_problems.append(list(sub_prob))

    return indices, sub_problems

# Function to merge subproblem solutions into the full problem
def merge_subproblem_solutions(sub_solutions, indices, n):
    merged_solution = [None] * n
    for sub_sol, sub_indices in zip(sub_solutions, indices):
        for idx, sol in zip(sub_indices, sub_sol.values()):
            merged_solution[idx] = sol
    return merged_solution

# Function to calculate final energy difference
def calculate_final_energy(merged_solution, weights):
    sum1 = sum(w for i, w in enumerate(weights) if merged_solution[i] == 0)
    sum2 = sum(w for i, w in enumerate(weights) if merged_solution[i] == 1)
    return abs(sum1 - sum2)

# Tabu search function for solving subproblems
def run_tabu_once(weights):
    bqm = generate_NPP_bqm(weights)
    sampler = TabuSampler()
    
    # Record start time
    start_time = time.time()

    # Solve using Tabu Sampler
    sampleset = sampler.sample(bqm)

    # Record execution time
    execution_time = time.time() - start_time

    best_solution = {k: int(v) for k, v in sampleset.first.sample.items()}  # Convert to Python int
    best_energy = sampleset.first.energy ** (0.5)

    return {"solution": best_solution, "E": best_energy, "time": execution_time}

# Function to run OUR with Tabu Sampler (OUR-tabu-40)
def our_solve_npp_tabu(weights, n_sub):
    timing_info = {}
    seed = random.randint(0, 1000000)  # Generate a random seed
    random.seed(seed)  # Set the seed for reproducibility

    start_time = time.time()

    # Generate subproblems
    step_start = time.time()
    indices, sub_problems = gen_sub_qubo_with_indices(weights, n_sub)
    timing_info["subproblem_generation"] = time.time() - step_start

    energy_optimals = []
    sub_solutions = []

    solve_times = []
    for k in range(n_sub):
        step_start = time.time()
        result = run_tabu_once(sub_problems[k])
        solve_times.append(time.time() - step_start)

        interm_energy = calculate_final_energy(result["solution"], sub_problems[k])
        energy_optimals.append(abs(interm_energy))
        sub_solutions.append(result["solution"])

    timing_info["subproblem_solving"] = sum(solve_times)
    timing_info["subproblem_solve_times"] = solve_times

    # Merge the subproblem solutions
    merged_solution = merge_subproblem_solutions(sub_solutions, indices, len(weights))
    final_diff = calculate_final_energy(merged_solution, weights)

    timing_info["total_execution_time"] = time.time() - start_time

    if abs(final_diff - calculate_final_energy(merged_solution, weights)) > 1e-6:
        print("WARNING: Final energy difference does not match the calculated difference!", flush=True)
        print(f"Final Energy Difference: {final_diff}, Calculated Difference: {calculate_final_energy(merged_solution, weights)}", flush=True)

    return {"E": final_diff, "solution": merged_solution, "random_seed": seed, "time": timing_info}

# Main OUR solve function, now parallelized across 3 threads for 5 runs
def our_solve_npp_parallel_tabu(weights, n_sub, num_runs=5, max_threads=3):
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(retry_on_exception, our_solve_npp_tabu, weights, n_sub) for _ in range(num_runs)]
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()  # Retrieve the result from each completed run
            results[f"run{i}"] = result  # Store the result under the appropriate run key
    
    return results

# Function to process and run the OUR-tabu-40 variant on the problems
def process_files_with_our_tabu(base_path):
    config = {"name": "OUR-tabu-40", "n_sub_factor": 40}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} with OUR-tabu-40", flush=True)

                # Load the JSON data
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    problem = data.get("problem", [])
                    n = len(problem)
                    n_sub = n // config["n_sub_factor"]

                # Run the OUR algorithm with Tabu in parallel using 3 threads
                results = our_solve_npp_parallel_tabu(problem, n_sub, num_runs=5, max_threads=3)

                # Save results to the JSON file
                data[config["name"]] = results

                # Write updated data back to JSON
                with open(file_path, "w") as json_output_file:
                    json.dump(data, json_output_file, indent=4)

                print(f"Processed {file_path} with OUR-tabu-40", flush=True)

base_path = "."

# Run the processing function
process_files_with_our_tabu(base_path)
