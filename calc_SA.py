import os
import json
import time
import dimod
from neal import SimulatedAnnealingSampler
import sys
from concurrent.futures import ThreadPoolExecutor

# Function to compute the absolute difference between the two sums based on the binary solution
def calculate_diff(solution, weights):
    sum1 = sum(w for i, w in enumerate(weights) if solution[i] == 0)
    sum2 = sum(w for i, w in enumerate(weights) if solution[i] == 1)
    return abs(sum1 - sum2)

# Function to create the correct QUBO formulation for the partitioning problem using Simulated Annealing
def sa_solve(weights, num=100):
    # Total sum of weights
    c = sum(weights)

    # Create the QUBO matrix
    Q = {}

    # Add off-diagonal terms: Q_ij = w_i * w_j
    for i in range(len(weights)):
        for j in range(len(weights)):
            Q[(i, j)] = weights[i] * weights[j]

    # Add diagonal terms: Q_ii = w_i * (w_i - c)
    for i in range(len(weights)):
        Q[(i, i)] = weights[i] * (weights[i] - c)

    # Create a BinaryQuadraticModel from the QUBO matrix
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

    # Initialize the Simulated Annealing Sampler
    sampler = SimulatedAnnealingSampler()

    # Store results of 5 runs
    results = {}
    for run_num in range(1, 6):  # Perform 5 runs
        start_time = time.time()

        # Sample solutions using SA
        sampleset = sampler.sample(bqm, num_reads=num)

        # Record time taken for this run
        sa_time = time.time() - start_time

        best_sample = [int(sampleset.first.sample[i]) for i in sorted(sampleset.first.sample.keys())]
        best_energy = (c ** 2 + 4 * sampleset.first.energy) ** (0.5)

        # Verify all samples to find the best solution to ensure if the first sample is the best
        real_best_energy = float('inf')
        for energy in sampleset.record.energy:
            # Compute the adjusted energy
            adjusted_energy = (c ** 2 + 4 * energy) ** (0.5)

            # Check if this is the best solution
            if adjusted_energy < real_best_energy:
                real_best_energy = adjusted_energy

        if abs(real_best_energy - best_energy) > 1e-6:
            print("WARNING: best_energy and real_best_energy do not match!", flush=True)
            print("best_energy", best_energy, flush=True)
            print("real_best_energy", real_best_energy, flush=True)

        # check with calculate_diff as well
        diff = calculate_diff(best_sample, weights)
        if abs(diff - best_energy) > 1e-6:
            print("WARNING: diff and best_energy do not match!", flush=True)
            print("diff", diff, flush=True)
            print("best_energy", best_energy, flush=True)

        # Store the results for this run
        results[f"run{run_num}"] = {
            "solution": best_sample,
            "E": best_energy,
            "time": sa_time
        }

    return results

# Function to process a single JSON file using Simulated Annealing
def process_file_with_sa(file_path, num=100):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
        problem = data.get("problem", [])

    print(f"Started processing {file_path}", flush=True)

    # Solve the problem 5 times using Simulated Annealing
    sa_results = sa_solve(problem, num=num)

    # Store all SA runs in the JSON file
    data["SA"] = sa_results

    # Save the updated data back to the file
    with open(file_path, "w") as json_output_file:
        json.dump(data, json_output_file, indent=4)

    print(f"Processed {file_path} with Simulated Annealing", flush=True)

# Function to process JSON files in parallel using 3 threads
def process_files_with_sa_parallel(base_path, num=100, max_workers=3):
    files_to_process = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda file_path: process_file_with_sa(file_path, num), files_to_process)

# Set the base path for JSON files and run in parallel with 3 threads
base_path = "."
process_files_with_sa_parallel(base_path, num=100, max_workers=3)
