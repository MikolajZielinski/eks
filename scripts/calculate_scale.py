import argparse
import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path


MODEL_TYPE = 'materials'
TIMESTAMP = 'black_background'

INPUT_PATH = Path(f"/var/home/mikolaj.zielinski/Git/genie/outputs/{MODEL_TYPE}/genie/{TIMESTAMP}/triangle_soup.ply")
INPUT_PATH_DEFORMED = Path(f"/var/home/mikolaj.zielinski/Git/genie/blender_neuraleditor/{MODEL_TYPE}/{MODEL_TYPE}_2.ply")
OUTPUT_PATH = Path(f"/var/home/mikolaj.zielinski/Git/genie/outputs/{MODEL_TYPE}/genie/{TIMESTAMP}/camera_path")

def main():

    if not INPUT_PATH.is_file() or not INPUT_PATH_DEFORMED.is_file():
        print(f"Input file not found: {INPUT_PATH} or {INPUT_PATH_DEFORMED}", file=sys.stderr)
        sys.exit(2)

    triangle_mesh = o3d.io.read_triangle_mesh(str(INPUT_PATH))
    verts = np.asarray(triangle_mesh.vertices)

    theta = -np.pi / 2
    R_z = np.array([
        [ np.cos(theta), -np.sin(theta), 0],
        [ np.sin(theta),  np.cos(theta), 0],
        [      0,               0,        1]
    ])

    # rotate points by -90 degrees around Z, then transpose back to (N, 3)
    verts = (R_z @ verts.T).T
    verts /= 3.0
    verts += np.array([0.5, 0.5, 0.5])

    print(f"Verts shape: {verts.shape}")

    triangle_mesh_deformed = o3d.io.read_triangle_mesh(str(INPUT_PATH_DEFORMED))

    print(verts.shape)
    print(triangle_mesh.faces.shape)

    # triangle_mesh = o3d.geometry.TriangleMesh()
    # triangle_mesh.vertices = o3d.utility.Vector3dVector(verts)
    

if __name__ == "__main__":
    main()