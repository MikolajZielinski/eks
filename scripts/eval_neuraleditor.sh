#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [-m model_type] [-t timestamp]"
    echo "  -m model type"
    echo "  -t timestamp"
    exit 1
}

while getopts ":m:t:h" opt; do
    case $opt in
        m) MODEL_TYPE="$OPTARG" ;;
        t) TIMESTAMP="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done
shift $((OPTIND -1))

echo "MODEL_TYPE=${MODEL_TYPE}"
echo "TIMESTAMP=${TIMESTAMP}"

# Generate tetrahedrons
genie-export tetrahedrons --load-config outputs/${MODEL_TYPE}/genie/${TIMESTAMP}/config.yml

# Scale up and prepare for deformation
python3 scripts/scale_up.py --model-type ${MODEL_TYPE} --timestamp ${TIMESTAMP}

# Run blender
cd blender_neuraleditor/${MODEL_TYPE}
/home/mikolaj/blender-3.3.21-linux-x64/blender -b ${MODEL_TYPE}.orig.blend -P mesh_deform.py
cd ../..

# Scale down
python3 scripts/scale_down.py --model-type ${MODEL_TYPE} --timestamp ${TIMESTAMP}

# Render images
genie-render dataset --load-config outputs/${MODEL_TYPE}/genie/${TIMESTAMP}/config.yml --output-path renders/${MODEL_TYPE}/${TIMESTAMP} --rendered-output-names rgb --image-format png --background-color black

# Calculate metrics
python3 scripts/calculate_metrics.py --model-type ${MODEL_TYPE} --timestamp ${TIMESTAMP}