#!/bin/bash
set -e

echo "📦 Installing GENIE package..."
cd /workspace/genie
pip install -e .
ns-install-cli
echo "✅ GENIE installation complete"
echo "🎉 Container ready!"

# Execute the original command
exec "$@"
