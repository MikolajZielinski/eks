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

ns-eval --load-config outputs/${MODEL_TYPE}/genie/${TIMESTAMP}/config.yml --output-path eval/${MODEL_TYPE}/${TIMESTAMP}/output.json --render-output-path eval/${MODEL_TYPE}/${TIMESTAMP}/renders