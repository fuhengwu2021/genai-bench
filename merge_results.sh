#!/bin/bash

# Script to merge results from multiple separate runs into one Excel file
# Usage: 
#   bash merge_results.sh                    # Default: H100-V3
#   bash merge_results.sh A100-v2           # For A100-v2
#   EXPERIMENT_PREFIX=A100-v2 bash merge_results.sh  # Alternative syntax

BASE_DIR="/home/fuhwu/workspace/benchmark/command-a-vision"
EXPERIMENT_BASE_DIR="${BASE_DIR}"

# Get experiment prefix from command line argument or environment variable
EXPERIMENT_PREFIX="${1:-${EXPERIMENT_PREFIX:-H100-V3}}"

# Create a merged experiment folder
MERGED_FOLDER="${BASE_DIR}/command-a-vision-${EXPERIMENT_PREFIX}-merged"
rm -rf "${MERGED_FOLDER}"
mkdir -p "${MERGED_FOLDER}"

echo "📊 Merging results from separate runs (prefix: ${EXPERIMENT_PREFIX})..."
echo ""

# Arrays to collect metadata from all scenarios
ALL_SCENARIOS=()
FIRST_METADATA_FILE=""

# Copy all JSON files from each experiment folder
for scenario in "i512" "i1024" "i2048"; do
    SOURCE_FOLDER="${BASE_DIR}/command-a-vision-${EXPERIMENT_PREFIX}-${scenario}"
    
    if [ -d "${SOURCE_FOLDER}" ]; then
        echo "📁 Copying results from ${scenario}..."
        
        # Copy all JSON files (metrics files) - these contain the actual data
        find "${SOURCE_FOLDER}" -name "*.json" -type f ! -name "experiment_metadata.json" | while read file; do
            # Get the filename
            filename=$(basename "$file")
            # Copy to merged folder
            cp "$file" "${MERGED_FOLDER}/${filename}"
        done
        
        # Collect metadata files for merging
        if [ -f "${SOURCE_FOLDER}/experiment_metadata.json" ]; then
            if [ -z "${FIRST_METADATA_FILE}" ]; then
                FIRST_METADATA_FILE="${SOURCE_FOLDER}/experiment_metadata.json"
            fi
            # Extract scenario name from metadata
            SCENARIO_NAME=$(grep -o '"I([0-9]*,[0-9]*)"' "${SOURCE_FOLDER}/experiment_metadata.json" | head -1 | tr -d '"')
            if [ -n "${SCENARIO_NAME}" ]; then
                ALL_SCENARIOS+=("${SCENARIO_NAME}")
            fi
        fi
    else
        echo "⚠️  Warning: ${SOURCE_FOLDER} not found, skipping..."
    fi
done

echo ""
echo "✅ Results copied to ${MERGED_FOLDER}"
echo ""

# Count JSON files
JSON_COUNT=$(find "${MERGED_FOLDER}" -name "*.json" -type f | wc -l)
echo "📊 Found ${JSON_COUNT} JSON files in merged folder"
echo ""

if [ "${JSON_COUNT}" -eq 0 ]; then
    echo "❌ No JSON files found! Please run the individual scenarios first."
    exit 1
fi

# Merge metadata files
if [ -n "${FIRST_METADATA_FILE}" ] && [ ${#ALL_SCENARIOS[@]} -gt 0 ]; then
    echo "📝 Merging experiment metadata..."
    
    # Create a temporary Python script to merge metadata
    cat > /tmp/merge_metadata.py << PYTHON_SCRIPT
import json
import sys
import os

# Get metadata files and scenarios from command line arguments
base_dir = sys.argv[1]
merged_folder = sys.argv[2]
experiment_prefix = sys.argv[3]
scenarios = sys.argv[4:]

metadata_files = [
    os.path.join(base_dir, f"command-a-vision-{experiment_prefix}-i512", "experiment_metadata.json"),
    os.path.join(base_dir, f"command-a-vision-{experiment_prefix}-i1024", "experiment_metadata.json"),
    os.path.join(base_dir, f"command-a-vision-{experiment_prefix}-i2048", "experiment_metadata.json")
]

# Filter out non-existent files
existing_files = [f for f in metadata_files if os.path.exists(f)]

if not existing_files:
    print("No metadata files found")
    sys.exit(1)

# Load the first metadata file as base
with open(existing_files[0], 'r') as f:
    merged_metadata = json.load(f)

# Merge traffic_scenario from all scenarios
merged_scenarios = []
for scenario in scenarios:
    if scenario and scenario not in merged_scenarios:
        merged_scenarios.append(scenario)

# Update the merged metadata
merged_metadata['traffic_scenario'] = merged_scenarios

# Merge num_concurrency from all metadata files (take the union)
all_concurrency = set(merged_metadata.get('num_concurrency', []))
for metadata_file in existing_files[1:]:
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            other_concurrency = metadata.get('num_concurrency', [])
            if isinstance(other_concurrency, list):
                all_concurrency.update(other_concurrency)
    except Exception as e:
        print(f"Warning: Could not read {metadata_file}: {e}")

merged_metadata['num_concurrency'] = sorted(list(all_concurrency))

# Save merged metadata
output_file = os.path.join(merged_folder, "experiment_metadata.json")
with open(output_file, 'w') as f:
    json.dump(merged_metadata, f, indent=4)

print(f"✅ Merged metadata saved to {output_file}")
print(f"   Scenarios: {merged_scenarios}")
print(f"   Concurrency levels: {sorted(list(all_concurrency))}")
PYTHON_SCRIPT

    # Run Python script with scenarios as arguments
    python3 /tmp/merge_metadata.py "${BASE_DIR}" "${MERGED_FOLDER}" "${EXPERIMENT_PREFIX}" "${ALL_SCENARIOS[@]}"
    
    # Clean up
    rm -f /tmp/merge_metadata.py
    
    echo ""
else
    echo "⚠️  Warning: No metadata files found to merge"
fi

echo "📊 Generating merged Excel report..."
echo ""

# Generate Excel report from merged folder
genai-bench excel \
  --experiment-folder "${MERGED_FOLDER}" \
  --excel-name "command-a-vision-${EXPERIMENT_PREFIX}-merged_summary" \
  --metric-percentile mean \
  --metrics-time-unit s

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Merged Excel report generated successfully!"
    echo "   Location: ${MERGED_FOLDER}/command-a-vision-${EXPERIMENT_PREFIX}-merged_summary.xlsx"
    echo ""
    echo "💡 Note: The merged folder contains all JSON files from individual runs."
    echo "   You can regenerate the Excel file anytime by running this script again."
else
    echo ""
    echo "❌ Failed to generate merged Excel report"
    echo "   Check the error messages above for details."
    exit 1
fi
