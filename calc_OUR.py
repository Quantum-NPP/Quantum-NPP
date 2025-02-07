import os
import json
import time
import random
import dimod
import neal
import minorminer
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming DWaveSampler setup (replace token with your actual token)
dwave_machine = DWaveSampler(token='XXX', solver='Advantage_system6.4')

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

def generate_NPP_bqm(weights):
    c = sum(weights)
    Q = {}
    
    for i in range(len(weights)):
        for j in range(len(weights)):
            Q[(i, j)] = 4 * weights[i] * weights[j]
    
    for i in range(len(weights)):
        Q[(i, i)] = 4 * weights[i] * (weights[i] - c)
    
    return dimod.BinaryQuadraticModel.from_qubo(Q=Q, offset=c**2)

def gen_sub_qubo_with_indices(weights, n_sub):
    indexed_weights = list(enumerate(weights))
    random.shuffle(indexed_weights)  # Shuffling with a set random seed for reproducibility
    
    indices, sub_problems = [], []
    per_sub = len(indexed_weights) // n_sub
    remainder = len(indexed_weights) % n_sub
    max_size = 0

    for i in range(n_sub):
        extra = 1 if i < remainder else 0
        start = i * per_sub + min(i, remainder)
        end = start + per_sub + extra
        sub_indices, sub_prob = zip(*indexed_weights[start:end])
        indices.append(list(sub_indices))
        sub_problems.append(list(sub_prob))
        max_size = max(max_size, len(sub_prob))
    
    return indices, sub_problems

def calculate_diff(solution, weights):
    solution = {s: (2*state-1) for s, state in solution.items()}
    spins = [solution[i] for i in range(len(weights))]
    return sum(w if s == +1 else -w for w, s in zip(weights, spins))

def list_sol(solution):
    return [solution[i] for i in range(len(solution))]

def standardize_sample(sample):
    """Ensure all solvers return results as dictionaries."""
    return {i: int(sample[i]) for i in range(len(sample))}

def sa_solve(weights, num=100):
    bqm = generate_NPP_bqm(weights)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num)
    best_sample = standardize_sample([int(sampleset.first.sample[i]) for i in sorted(sampleset.first.sample.keys())])
    best_energy = sampleset.first.energy ** 0.5
    return best_energy, best_sample

def compile_original_solution(sub_solutions, sub_indices):
    n = sum(1 for sublist in sub_indices for idx in sublist if idx != -1)
    original_solution = [None] * n
    for solution, indices in zip(sub_solutions, sub_indices):
        for sol, index in zip(solution.values(), indices):
            if index != -1:
                original_solution[index] = sol
    return original_solution

def calculate_diff_with_list(solution, weights):
    return sum(w if s == +1 else -w for w, s in zip(weights, solution))

def qa_solve(weights):
    bqm = generate_NPP_bqm(weights)
    embedding = minorminer.find_embedding(bqm.quadratic, dwave_machine.edgelist)
    sampler = FixedEmbeddingComposite(dwave_machine, embedding)
    
    # Start the sampling
    results = sampler.sample(bqm, num_reads=100)
    
    # Extract the best sample
    best_sample = standardize_sample([int(results.first.sample[i]) for i in sorted(results.first.sample.keys())])
    best_energy = results.first.energy ** 0.5
    
    # Extract timing information
    qpu_timing = results.info['timing']
    
    return best_energy, best_sample, qpu_timing

# Run OUR algorithm once (used in retries)
def our_solve_npp(weights, n_sub, option="hybrid"):
    timing_info = {}
    
    seed = random.randint(0, 1000000)  # Generate a random seed if not provided
    random.seed(seed)  # Set the seed for reproducibility

    start_time = time.time()

    # Step 1: Generate subproblems
    step_start = time.time()
    indices, sub_problems = gen_sub_qubo_with_indices(weights, n_sub)
    timing_info["subproblem_generation"] = time.time() - step_start

    energy_optimals = []
    sub_solutions = []

    qpu_timings = []

    # Initialize QPU embedding for QA/hybrid if needed
    if option in ["hybrid", "QA"]:
        step_start = time.time()
        embedding = minorminer.find_embedding(generate_NPP_bqm(sub_problems[0]).quadratic, dwave_machine.edgelist)
        fixed_sampler = FixedEmbeddingComposite(dwave_machine, embedding)
        timing_info["embedding_time"] = time.time() - step_start

    # Step 2: Solve subproblems
    solve_times = []
    for k in range(n_sub):
        step_start = time.time()
        bqm = generate_NPP_bqm(sub_problems[k])
        if option == "QA":
            s_results = fixed_sampler.sample(bqm, num_reads=100)
            s_opt_result = standardize_sample(s_results.first.sample)
            qpu_timing = s_results.info['timing']  # Capture QPU timing
            qpu_timings.append(qpu_timing)
        elif option == "SA":
            _, s_opt_result = sa_solve(sub_problems[k])
        elif option == "hybrid":
            s_results = fixed_sampler.sample(bqm, num_reads=100)
            s_opt_result = standardize_sample(s_results.first.sample)
            qpu_timing = s_results.info['timing']  # Capture QPU timing
            qpu_timings.append(qpu_timing)
        solve_times.append(time.time() - step_start)

        interm = calculate_diff(s_opt_result, sub_problems[k])
        energy_optimals.append(abs(interm))

        if interm >= 0:
            sub_solutions.append(s_opt_result)
        else:
            sub_solutions.append({i: 1 if s == 0 else 0 for i, s in s_opt_result.items()})

    timing_info["subproblem_solving"] = sum(solve_times)
    timing_info["subproblem_solve_times"] = solve_times

    # Step 3: Merge sub-solutions using SA or QA
    step_start = time.time()
    if option == "QA":
        bqm = generate_NPP_bqm(energy_optimals)
        results = fixed_sampler.sample(bqm, num_reads=100)
        final_energy = results.first.energy ** 0.5
        intermediate_solution = standardize_sample([int(results.first.sample[i]) for i in sorted(results.first.sample.keys())])
    elif option == "SA" or option == "hybrid":
        final_energy, intermediate_solution = sa_solve(energy_optimals)
    timing_info["merging_time"] = time.time() - step_start

    # Adjust sub-solutions based on intermediate solution
    for j in range(len(intermediate_solution)):
        if intermediate_solution[j] == 0:
            for k in sub_solutions[j]:
                sub_solutions[j][k] = 1 if sub_solutions[j][k] == 0 else 0

    # Compile the final solution
    step_start = time.time()
    global_solution = compile_original_solution(sub_solutions, indices)
    timing_info["final_solution_compilation"] = time.time() - step_start

    final_diff = abs(calculate_diff_with_list(global_solution, weights))
    timing_info["total_execution_time"] = time.time() - start_time

    if abs(final_diff - final_energy) > 1e-6:
        print("WARNING: final_diff and final_energy do not match!", flush=True)
        print("final_diff", final_diff, flush=True)
        print("final_energy", final_energy, flush=True)

    # Include QPU timings if any
    if qpu_timings:
        timing_info["qpu_timings"] = qpu_timings

    return {"E": final_diff, "solution": global_solution, "random_seed": seed, "time": timing_info}

# Main OUR solve function, now parallelized across 3 threads for 5 runs
def our_solve_npp_parallel(weights, n_sub, option="hybrid", num_runs=5, max_threads=3):
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(retry_on_exception, our_solve_npp, weights, n_sub, option) for _ in range(num_runs)]
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()  # Retrieve the result from each completed run
            results[f"run{i}"] = result  # Store the result under the appropriate run key
    
    return results

# Function to process and run the different configurations (hybrid, SA-40, QA-40)
def process_files_with_our(base_path):
    configs = [
        {"name": "OUR-hybrid-20", "option": "hybrid", "n_sub_factor": 20},
        {"name": "OUR-hybrid-40", "option": "hybrid", "n_sub_factor": 40},
        {"name": "OUR-hybrid-60", "option": "hybrid", "n_sub_factor": 60},
        {"name": "OUR-hybrid-80", "option": "hybrid", "n_sub_factor": 80},
        {"name": "OUR-SA-40", "option": "SA", "n_sub_factor": 40},
        {"name": "OUR-QA-40", "option": "QA", "n_sub_factor": 40}
    ]

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} with OUR algorithm", flush=True)

                # Load the JSON data
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    problem = data.get("problem", [])
                    n = len(problem)

                for config in configs:
                    n_sub = n // config["n_sub_factor"]

                    # Run the OUR algorithm with retries in parallel using 3 threads
                    results = our_solve_npp_parallel(problem, n_sub, config["option"], num_runs=5, max_threads=3)

                    # Save results to the JSON file
                    data[config["name"]] = results

                # Write updated data back to JSON
                with open(file_path, "w") as json_output_file:
                    json.dump(data, json_output_file, indent=4)

                print(f"Processed {file_path} with OUR algorithm", flush=True)

base_path = "."

# Run the processing function
process_files_with_our(base_path)
