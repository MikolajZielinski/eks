import os;
import torch;

# ### ### ### ### ###

# torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib');
# os.add_dll_directory(torch_lib_path);

# os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin");

import optix_knnx as PyOptiXKNN;

# ### ### ### ### ###

print("CPyOptiXKNN initialization in progress... .");
py_optix_knn = PyOptiXKNN.CPyOptiXKNN(110.3449);
print("CPyOptiXKNN initialization done... .");

number_of_Gaussians = 1000000;

m = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32, device="cuda") + (torch.rand(number_of_Gaussians, 3, dtype=torch.float32, device="cuda") * 2.0);
s = torch.Tensor.repeat(torch.tensor([0.01, 0.005, 0.0025], dtype=torch.float32, device="cuda"), number_of_Gaussians, 1);
q = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device="cuda") + (torch.rand(number_of_Gaussians, 4, dtype=torch.float32, device="cuda") * 2.0);

print("Fit in progress... .");
py_optix_knn.Fit(m, s, q);
print("Fit done... .");

number_of_queried_points = 10000000;
K = 16;

queried_points = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32, device="cuda") + (torch.rand(number_of_queried_points, 3, dtype=torch.float32, device="cuda") * 2.0);

print("KNeighbors in progress... .");
indices, distances_squared = py_optix_knn.KNeighbors(queried_points, K);
print("KNeighbors done... .");

number_of_neighbors = (indices != -1).sum(0);

print("Minimum number of neighbors:", torch.min(number_of_neighbors).item());
print("Average number of neighbors:", torch.mean(number_of_neighbors.to(dtype=torch.float32)).item());
print("Maximum number of neighbors:", torch.max(number_of_neighbors).item());



