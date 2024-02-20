import h5py
import torch
from jacobi_pytorch import jacobi

if __name__ == "__main__":
    n = 1000
    grid = torch.rand(n, n)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0.0
    grid = grid.cuda()
    for _ in range(n):
        grid = jacobi(grid)

    grid_np = grid.cpu().numpy()
    with h5py.File("jacobi_pytorch.hdf5","w") as f:
        dset=f.create_dataset("Grid",  data = grid_np)