import numpy as np

def read_ply(ply_file_path) -> dict:

    with open(ply_file_path, "rb") as f:
        # Parse header
        properties = []
        vertex_count = 0
        
        while True:
            line = f.readline().decode("utf-8").strip()
            if line == "end_header":
                break
            
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            
            if line.startswith("property"):
                parts = line.split()
                # format: property <type> <name>
                dtype_str = parts[1]
                name = parts[2]
                
                if dtype_str == "float":
                    np_dtype = "f4"
                elif dtype_str == "uchar":
                    np_dtype = "u1"
                else:
                    # Handle other types if necessary
                    mapping = {
                        "double": "f8", "int": "i4", "uint": "u4", 
                        "short": "i2", "ushort": "u2", "char": "i1"
                    }
                    np_dtype = mapping.get(dtype_str, "f4")
                
                properties.append((name, np_dtype))

        # Read binary data
        dtype = np.dtype(properties)
        gaussians = np.fromfile(f, dtype=dtype, count=vertex_count)

        print(f"Loaded {gaussians['x'].shape[0]} 3D points from PLY")

    return gaussians