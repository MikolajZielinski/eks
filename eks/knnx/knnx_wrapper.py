import ctypes
import os
import torch
import numpy as np

class OptiXKNN:
    def __init__(self, chi_square_squared_radius: float):
        # Load the library
        lib_path = os.path.join(os.path.dirname(__file__), "knnx_core.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find knnx_core.so at {lib_path}. Please build the library first.")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define argument types
        self.lib.CreateKNN.argtypes = [ctypes.c_float, ctypes.c_char_p]
        self.lib.CreateKNN.restype = ctypes.c_void_p
        
        self.lib.DestroyKNN.argtypes = [ctypes.c_void_p]
        self.lib.DestroyKNN.restype = None
        
        self.lib.Fit.argtypes = [
            ctypes.c_void_p, # knn_ptr
            ctypes.c_void_p, # m
            ctypes.c_void_p, # s
            ctypes.c_void_p, # q
            ctypes.c_int     # number_of_Gaussians
        ]
        self.lib.Fit.restype = None
        
        self.lib.KNeighbors.argtypes = [
            ctypes.c_void_p, # knn_ptr
            ctypes.c_void_p, # queried_points
            ctypes.c_int,    # number_of_points
            ctypes.c_int,    # K
            ctypes.c_void_p, # indices
            ctypes.c_void_p  # distances_squared
        ]
        self.lib.KNeighbors.restype = None
        
        # Initialize
        ptx_path = os.path.join(os.path.dirname(__file__), "shaders.ptx")
        if not os.path.exists(ptx_path):
             raise FileNotFoundError(f"Could not find shaders.ptx at {ptx_path}. Please build the library first.")
             
        self.knn_ptr = self.lib.CreateKNN(chi_square_squared_radius, ptx_path.encode('utf-8'))
        
    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'knn_ptr') and self.knn_ptr:
            self.lib.DestroyKNN(self.knn_ptr)
            
    def fit(self, m: torch.Tensor, s: torch.Tensor, q: torch.Tensor):
        if m.device.type != 'cuda' or s.device.type != 'cuda' or q.device.type != 'cuda':
            raise ValueError("All input tensors must be on CUDA device")
            
        number_of_Gaussians = m.shape[0]
        
        self.lib.Fit(
            self.knn_ptr,
            ctypes.c_void_p(m.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            number_of_Gaussians
        )
        
    def kneighbors(self, queried_points: torch.Tensor, K: int):
        if queried_points.device.type != 'cuda':
            raise ValueError("Queried points must be on CUDA device")
            
        number_of_points = queried_points.shape[0]
        
        indices = torch.empty((K, number_of_points), dtype=torch.int32, device=queried_points.device)
        distances_squared = torch.empty((K, number_of_points), dtype=torch.float32, device=queried_points.device)
        
        self.lib.KNeighbors(
            self.knn_ptr,
            ctypes.c_void_p(queried_points.data_ptr()),
            number_of_points,
            K,
            ctypes.c_void_p(indices.data_ptr()),
            ctypes.c_void_p(distances_squared.data_ptr())
        )
        
        return indices, distances_squared
