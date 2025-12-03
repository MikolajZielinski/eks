import argparse
import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path


def get_model_and_timestamp():
    parser = argparse.ArgumentParser(description="Scale and center a mesh")
    parser.add_argument("--model-type", default='chair', dest="model_type")
    parser.add_argument("--timestamp", default='new_denisfy', dest="timestamp")
    args = parser.parse_args()

    return args.model_type, args.timestamp

def main():

    model_type, timestamp = get_model_and_timestamp()

    input_path = Path(f"blender_neuraleditor/{model_type}/{model_type}_deformed.ply")
    output_path = Path(f"outputs/{model_type}/genie/{timestamp}/camera_path")
    data_path = Path(f"blender_neuraleditor/{model_type}/views_test")

    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    triangle_mesh = o3d.io.read_triangle_mesh(str(input_path))

    theta = np.pi / 2
    R_z = np.array([
        [ np.cos(theta), -np.sin(theta), 0],
        [ np.sin(theta),  np.cos(theta), 0],
        [      0,               0,        1]
    ])

    # rotate points by -90 degrees around Z, then transpose back to (N, 3)
    verts = np.asarray(triangle_mesh.vertices)
    verts = (R_z @ verts.T).T
    verts /= 3.0
    verts += np.array([0.5, 0.5, 0.5])

    print(f"Verts shape: {verts.shape}")

    triangle_mesh.vertices = o3d.utility.Vector3dVector(verts)
    
    output_path.mkdir(parents=True, exist_ok=True)
    len_data = len(os.listdir(data_path)) - 1

    for i in range(len_data):
        o3d.io.write_triangle_mesh(str(output_path / f"{i:05d}.ply"), triangle_mesh, write_ascii=True)
        
if __name__ == "__main__":
    main()