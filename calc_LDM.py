import os
import json
import time
import heapq

# Function to apply Largest Differencing Method (LDM) using a heap and reconstruct the solution
def largest_differencing_method(weights):
    # Convert weights into a max heap by using negative values because Python's heapq is a min-heap.
    # Each element is a tuple (-weight, [original element]).
    heap = [(-weight, [weight]) for weight in weights]
    heapq.heapify(heap)
    
    # The heap will contain tuples of (-weight, [list of original elements])
    while len(heap) > 1:
        # Pop the two largest weights (remember, we're working with negative values, so this gives us the largest).
        largest_weight, largest_elems = heapq.heappop(heap)
        second_largest_weight, second_largest_elems = heapq.heappop(heap)
        
        # Calculate the difference between the two largest weights
        difference_weight = largest_weight - second_largest_weight
        
        # Combine the elements from the two partitions
        combined_elems = largest_elems + [-x for x in second_largest_elems]
        
        # Push the new combined weight and elements back into the heap
        heapq.heappush(heap, (difference_weight, combined_elems))
    
    # The last remaining element in the heap contains the final partition difference and corresponding elements.
    final_diff, final_elems = heap[0]
    
    # Split the elements into two partitions: positive values in one, negative values in the other.
    partition1 = [x for x in final_elems if x > 0]
    
    # Reconstruct the solution: binary array indicating which partition each item belongs to.
    solution = []
    partition1_copy = partition1.copy()  # Copy partition1 to track assignments
    for w in weights:
        if w in partition1_copy:
            partition1_copy.remove(w)
            solution.append(1)  # Assign to partition 1
        else:
            solution.append(0)  # Assign to the other partition (0)
    
    return solution, -final_diff

base_path = "."

# Define the function to iterate through files, apply LDM, and save results with time tracking
def process_files_with_ldm(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    problem = data.get("problem", [])
                    
                    # Apply Largest Differencing Method (LDM) and track time
                    start_time = time.time()
                    ldm_solution, ldm_E = largest_differencing_method(problem)
                    ldm_time = time.time() - start_time  # Time in seconds
                    
                    # Add LDM solution, energy, and time to the JSON
                    data["LDM"] = {
                        "run1": {
                            "solution": ldm_solution,
                            "E": ldm_E,
                            "time": ldm_time
                        }
                    }
                    
                    # Save the updated data back to the file
                    with open(file_path, "w") as json_output_file:
                        json.dump(data, json_output_file, indent=4)
                    
                    print(f"Processed {file_path} with LDM")

# Call the function to process files
process_files_with_ldm(base_path)
