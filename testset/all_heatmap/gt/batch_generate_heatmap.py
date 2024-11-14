import torch
import numpy as np
import os
from multiprocessing import Pool
import fire

def create_heatmap_matrix(batch_coords, mask_val=1e6):
    gt_prior = {
        500: 
        np.array([4.40078125e-01, 2.56265625e-01, 1.32750000e-01, 7.32656250e-02,
        4.08125000e-02, 2.35937500e-02, 1.34062500e-02, 7.75000000e-03,
        4.48437500e-03, 2.73437500e-03, 1.78125000e-03, 1.18750000e-03,
        6.87500000e-04, 3.75000000e-04, 3.75000000e-04, 1.87500000e-04,
        7.81250000e-05, 1.56250000e-05, 4.68750000e-05, 1.56250000e-05,
        4.68750000e-05, 3.12500000e-05, 1.56250000e-05, 1.56250000e-05]),
        1000:
        np.array([4.37554687e-01, 2.54718750e-01, 1.37671875e-01, 7.41093750e-02,
        3.97890625e-02, 2.35156250e-02, 1.32265625e-02, 7.45312500e-03,
        4.73437500e-03, 3.00781250e-03, 1.59375000e-03, 1.08593750e-03,
        5.62500000e-04, 2.96875000e-04, 2.65625000e-04, 1.71875000e-04,
        1.01562500e-04, 4.68750000e-05, 1.56250000e-05, 3.12500000e-05,
        2.34375000e-05, 7.81250000e-06, 1.56250000e-05]),
        10000:
        np.array([4.4175625e-01, 2.5409375e-01, 1.3292500e-01, 7.1950000e-02,
        3.9518750e-02, 2.3750000e-02, 1.4143750e-02, 8.0937500e-03,
        4.9125000e-03, 3.3312500e-03, 1.8437500e-03, 1.1125000e-03,
        8.3750000e-04, 5.5625000e-04, 3.7500000e-04, 2.6250000e-04,
        1.8125000e-04, 8.7500000e-05, 6.8750000e-05, 5.0000000e-05,
        5.0000000e-05, 2.5000000e-05, 2.5000000e-05, 6.2500000e-06,
        1.2500000e-05, 6.2500000e-06, 6.2500000e-06, 6.2500000e-06,
        6.2500000e-06, 6.2500000e-06])
    }
    B, n, _ = batch_coords.shape

    prior = gt_prior[n]
    prior = prior / prior.sum()
    prior = torch.from_numpy(prior).float()
    topk = len(prior)
    batch_coords = torch.from_numpy(batch_coords).float()
    expanded_coords = batch_coords.unsqueeze(2)
    tiled_coords = batch_coords.unsqueeze(1)
    dist_matrix = torch.norm(expanded_coords - tiled_coords, dim=-1)
    
    eye = torch.eye(n, device=batch_coords.device).unsqueeze(0) * mask_val
    dist_matrix += eye

    heatmap = torch.zeros_like(dist_matrix)
    topk_values, topk_indices = torch.topk(dist_matrix, topk, largest=False, dim=-1)
    topk_values = prior.unsqueeze(0).unsqueeze(0).expand(B, n, topk)

    heatmap.scatter_(2, topk_indices, topk_values)  
    return heatmap.cpu().numpy()


def read_tsp_file(file_path, N):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split(" output ")
        coords_flat = np.array(parts[0].split(), dtype=np.float32)
        data.append(coords_flat[:2*N])
    data = np.array(data).reshape(-1, N, 2)
    return data

def write_heatmap_to_file(args):
    heatmap_matrix, output_file, N = args
    with open(output_file, 'w') as out_file:
        out_file.write(f"{N}\n")
        for row in heatmap_matrix:
            out_file.write(' '.join(map(str, row)) + '\n')

def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)

def process_batch(batch_coords, N, batch_size):
    for i in range(0, len(batch_coords), batch_size):
        batch = batch_coords[i:i+batch_size]
        heatmap_matrices = create_heatmap_matrix(batch)
        
        args = [(heatmap_matrix, f"./heatmap/tsp{N}/heatmaptsp{N}_{i+j}.txt", N) 
                for j, heatmap_matrix in enumerate(heatmap_matrices)]
        
        with Pool() as pool:
            pool.map(write_heatmap_to_file, args)

def process_tsp_data(N):
    file_path = f"../../tsp{N}_test_concorde.txt"

    batch_coords = read_tsp_file(file_path, N)

    os.makedirs(f"./heatmap/tsp{N}", exist_ok=True)
    try:
        process_batch(batch_coords, N, len(batch_coords))
        print("Processed in a single batch.")

    except RuntimeError as e:
        print(f"Error occurred: {e}. Please enter a valid batch size (must be a power of two and less than or equal to {len(batch_coords)}):")
        
        while True:
            try:
                batch_size = int(input("Enter batch size: "))
                if not is_power_of_two(batch_size) or batch_size > len(batch_coords):
                    raise ValueError
                
                process_batch(batch_coords, N, batch_size)
                break

            except ValueError:
                print("Invalid batch size. It must be a power of two and less than or equal to the total number of batches.")
            except RuntimeError as e:
                print(f"Error occurred again: {e}. Please enter a smaller batch size:")


def main(N):
    process_tsp_data(N)

if __name__ == "__main__":
    fire.Fire(main)