#!/bin/bash

# Run genai-bench with log directory specified for H100
genai-bench benchmark \
  --api-backend oci-cohere \
  --api-base "https://ppe.inference.generativeai.us-chicago-1.oci.oraclecloud.com" \
  --config-file /home/fuhwu/.oci/config --profile command_a_20251119 \
  --api-model-name "command-a-vision-H100" \
  --model-tokenizer "/home/fuhwu/workspace/benchmark/command-a-vision/tokenizer" \
  --experiment-folder-name "command-a-vision-H100-V3" \
  --log-dir "/home/fuhwu/workspace/benchmark/command-a-vision/logs-h100-v3" \
  --task image-text-to-text \
  --dataset-config /home/fuhwu/workspace/benchmark/genai-bench/examples/dataset_configs/config_llava-bench-in-the-wild.json \
  --max-time-per-run 15 \
  --max-requests-per-run 300 \
  --server-engine "vLLM" \
  --server-gpu-type "H100" \
  --server-version "latest" \
  --server-gpu-count 4 \
        --traffic-scenario 'I(512,512)' \
        --traffic-scenario 'I(1024,1024)' \
        --traffic-scenario 'I(2048,2048)' \
  --additional-request-params '{
    "compartmentId": "ocid1.compartment.oc1..aaaaaaaabma2uwi3rcrlx5qxsihcr2k4ehf7jxer6p6c6ngga2zlhkgir3ka",
    "endpointId": "ocid1.generativeaiendpoint.oc1.us-chicago-1.amaaaaaabgjpxjqaejokbz4b3g2idcqv4ger7p7tdtutjbd52zytuqfnrhsa",
    "servingType": "DEDICATED"
  }' --num-workers 4 --master-port 5558 --spawn-rate 10 --heartbeat-timeout 120


