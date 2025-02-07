import json
import random

# Function to generate a single NPP (Number Partitioning Problem)
def generate_npp(n):
    lower_bound = 5 * n
    upper_bound = 10 * n
    weights = [random.randint(lower_bound, upper_bound) for _ in range(n)]
    return weights

# Values for n
n_values = [325, 500, 675, 850, 1025, 1200]

# Generating 10 NPPs for each n and saving them in separate json files
for n in n_values:
    for i in range(1, 11):  # generate 10 problems for each n
        problem = generate_npp(n)
        file_name = f"NPP_n{str(n).zfill(4)}_problem{str(i).zfill(2)}.json"
        data = {"problem": problem}
        with open(f"./data/{file_name}", "w") as json_file:
            json.dump(data, json_file)
