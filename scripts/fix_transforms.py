import json
import sys
import numpy as np

MODEL_TYPE = 'materials'

DATA_PATH = f"/var/home/mikolaj.zielinski/Git/genie/data/nerf_synthetic/{MODEL_TYPE}/transforms_test_neuraleditor.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def fix_matrix(mat):
    """
    Fix camera poses by applying a ±90° rotation around Z axis.
    Use +90° or -90° depending on which direction corrects your renders.
    """
    M = np.array(mat, dtype=np.float64)

    # Choose the angle:
    # If result is rotated the wrong way, flip the sign.
    theta = -np.pi / 2      # -90° around Z
    # theta = np.pi / 2   # (uncomment if opposite direction is needed)

    # Rotation around Z axis (right-handed)
    R_z = np.array([
        [ np.cos(theta), -np.sin(theta), 0, 0],
        [ np.sin(theta),  np.cos(theta), 0, 0],
        [      0,               0,        1, 0],
        [      0,               0,        0, 1]
    ])

    # Apply to the pose: M_fixed = R_z * M
    M_fixed = R_z @ M
    return M_fixed.tolist()

def main(input_path):
    data = load_json(input_path)

    for frame in data["frames"]:
        frame["transform_matrix"] = fix_matrix(frame["transform_matrix"])

    out_path = input_path.replace("_neuraleditor.json", ".json")
    save_json(out_path, data)
    print("Saved corrected file:", out_path)

if __name__ == "__main__":

    main(DATA_PATH)
