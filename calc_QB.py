import time
from dwave.system import DWaveSampler, AutoEmbeddingComposite
import dimod
import hybrid
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

class DwaveQuantumSampler():
    def __init__(self, token, **kwargs):
        self.parameters = {}
        self.properties = {}
        self.name = 'DWave_Quantum_Sampler'
        
        # Tracking variables
        self._counter = 0
        self._num_reads = kwargs.get('num_reads_subsampler', 1)
        
        # Arrays to store BQMs, timing, energies, and timing info
        self._bqms = []
        self._sample_times = []
        self._sample_energies = []
        self._sample_timing_info = []
        
        # Initialize the D-Wave sampler
        self._sampler_machine = DWaveSampler(token=token)
        self._emb = AutoEmbeddingComposite(self._sampler_machine)

    def sample(self, bqm: dimod.BinaryQuadraticModel):
        """The actual sample function. Takes a BQM (Binary Quadratic Model) and returns the response."""
        self._counter += 1
        
        # Store the BQM
        self._bqms.append(bqm.to_serializable())
        
        # Record the time for sampling
        tic = time.perf_counter()  # Start time
        response = self._emb.sample(bqm, num_reads=self._num_reads, label="Sample Problem")
        toc = time.perf_counter()  # End time
        
        # Save the sample time, energy, and timing info to arrays
        sample_time = toc - tic
        self._sample_times.append(sample_time)
        self._sample_energies.append(response.first.energy)
        self._sample_timing_info.append(response.info['timing'])
        
        return response

    @property
    def counter(self):
        return self._counter

    @property
    def bqms(self):
        """Retrieve the array of sampled BQMs."""
        return self._bqms

    @property
    def sample_times(self):
        """Retrieve the array of sample times."""
        return self._sample_times

    @property
    def sample_energies(self):
        """Retrieve the array of sample energies."""
        return self._sample_energies

    @property
    def sample_timing_info(self):
        """Retrieve the array of sample timing info."""
        return self._sample_timing_info

    def reset_counter(self):
        self._counter = 0

    def reset(self):
        """Reset all internal counters and arrays."""
        self.reset_counter()
        self._bqms = []
        self._sample_times = []
        self._sample_energies = []
        self._sample_timing_info = []
        
    def set_num_reads(self, num_reads):
        """Set the number of reads for the sampler."""
        self._num_reads = num_reads

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

## Simplified QBSolv Routine
def SimplifiedQbsolv(max_iter=1, max_time=None, convergence=3,
                     energy_threshold=None, max_subproblem_size=40, qpu_sampler=None):
    """Races a Tabu solver and a QPU-based sampler of flip-energy-impact induced
    subproblems.

    For arguments description see: :class:`~hybrid.reference.kerberos.Kerberos`.
    """

    energy_reached = None
    if energy_threshold is not None:
        energy_reached = lambda en: en <= energy_threshold

    workflow = hybrid.Loop(
        hybrid.Race(
            hybrid.InterruptableTabuSampler(), ## Zeile entfernen um Tabusearch als Parallel-run auszuschließen.
            hybrid.EnergyImpactDecomposer(
                size=max_subproblem_size, rolling=True, rolling_history=0.15)
            | hybrid.QPUSubproblemAutoEmbeddingSampler(qpu_sampler=qpu_sampler)
            | hybrid.SplatComposer()
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True),
        max_iter=max_iter, max_time=max_time,
        convergence=convergence, terminate=energy_reached)

    return workflow

def calculate_diff_with_list(solution, weights):
    return abs(sum(w if s == +1 else -w for w, s in zip(weights, solution)))

def calc_QBSolv(weights, qpu_token=None):
    """Runs the simplified QBSolv routine and returns results, including timing and energies."""
    
    # Initialize DWave Quantum Sampler with provided token
    qpu_sampler = DwaveQuantumSampler(token=qpu_token)

    # Generate Binary Quadratic Model (BQM) for NPP
    bqm = generate_NPP_bqm(weights)

    # Initialize timing
    tic = time.perf_counter()

    # Create the QBSolv hybrid workflow with the parameters
    workflow = SimplifiedQbsolv(
        qpu_sampler=qpu_sampler  # Pass the auto-embedding composite from our DWave Quantum Sampler
    )

    # Execute the workflow with the generated BQM
    init_state = hybrid.State.from_sample(hybrid.random_sample(bqm), bqm)
    result = workflow.run(init_state).result()

    # Record the end time
    toc = time.perf_counter()
    total_time = toc - tic

    # Extract final solution and minimum energy
    solution = [int(result.samples.first.sample[i]) for i in sorted(result.samples.first.sample.keys())]
    min_energy = result.samples.first.energy ** (0.5)

    # Calculate the difference based on the solution
    diff = calculate_diff_with_list(solution, weights)

    if abs(min_energy - diff) > 1e-6:
        print("WARNING: min_energy and diff do not match!", flush=True)
        print("min_energy", min_energy, flush=True)
        print("diff", diff, flush=True)

    # Prepare the output dictionary
    output = {
        "solution": solution,
        "E": min_energy,
        "time": {
            "total_execution_time": total_time,
            "sample_times": qpu_sampler.sample_times,
            "sample_timing_info": qpu_sampler.sample_timing_info
        },
        "bqms_sampled": qpu_sampler.bqms,
        "sample_times": qpu_sampler.sample_times,
        "sample_energies": qpu_sampler.sample_energies,
        "qpu_counter": qpu_sampler.counter
    }

    return output

# Retry mechanism with exception handling
def retry_on_exception(func, *args, retries=100, wait_time=10, **kwargs):
    """Retries the function call if an exception occurs."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Exception occurred: {e}. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{retries})")
                sleep(wait_time)
            else:
                print(f"Failed after {retries} retries.")
                raise

# Function to process and run 5 parallel QBSolv executions for each problem
def process_files_with_qbsolv(base_path, qpu_token=None):
    # Define the number of parallel runs and threads
    num_runs = 5
    max_threads = 3

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} with QBSolv", flush=True)

                # Load the JSON data
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    problem = data.get("problem", [])
                
                # Parallel execution using ThreadPoolExecutor
                results = {}
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    futures = [executor.submit(retry_on_exception, calc_QBSolv, problem, qpu_token) for _ in range(num_runs)]
                    
                    # Collect results as futures complete
                    for i, future in enumerate(as_completed(futures), 1):
                        results[f"run{i}"] = future.result()

                # Save results back to the JSON file
                data["QBSolv"] = results

                # Write updated data back to JSON
                with open(file_path, "w") as json_output_file:
                    json.dump(data, json_output_file, indent=4)

                print(f"Processed {file_path} with QBSolv", flush=True)

# Set the path to the folder containing JSON problem files and QPU token
base_path = "."
qpu_token = "xxx"

# Run the processing function
process_files_with_qbsolv(base_path, qpu_token)