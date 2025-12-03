import argparse
import os
import sys
import numpy as np
import open3d as o3d


def main():
    default_in = "/var/home/mikolaj.zielinski/Git/genie/blender_neuraleditor/materials/materials_01.npy"
    parser = argparse.ArgumentParser(description="Convert a .npy of vertices to a .ply containing only vertices (points).")
    parser.add_argument("input", nargs="?", default=default_in, help="Input .npy file (default: %(default)s)")
    parser.add_argument("output", nargs="?", help="Output .ply file (default: replace .npy with .ply)")
    args = parser.parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(2)

    out = args.output or (os.path.splitext(inp)[0] + ".ply")

    verts = np.load(inp)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)

    success = o3d.io.write_point_cloud(out, pcd, write_ascii=True)
    if not success:
        print("Failed to write PLY.", file=sys.stderr)
        sys.exit(3)
    print(f"Wrote {len(verts)} vertices to: {out}")

if __name__ == "__main__":
    main()