#!/bin/bash
# Run all post-inference evaluation steps using the released flat data layout.
# Assumes run_tool_benchmark_all_released.py has already completed (result_*.json files exist).
#
# Released layout (fdb_v3_data_released):
#     {example_id}_{speaker_id}/input.wav
#     {example_id}_{speaker_id}/metadata.json
#     {example_id}_{speaker_id}/result_{provider}.json   (after inference)
#
# Usage:
#   conda activate fdb
#   bash run_all_evaluations_released.sh
#   bash run_all_evaluations_released.sh --skip-existing

set -e
cd "$(dirname "$0")"

BENCHMARK="benchmark_data_v2.json"
RESULTS_DIR="fdb_v3_data_released"
SKIP_FLAG="${1:-}"

# Edit this list to include the providers you have run inference for:
PROVIDERS=("gpt_realtime" "gemini2_5" "grok" "gemini3_1" "ultravox" "cascaded")

echo "============================================"
echo "Multi-Step Tool Benchmark: All Evaluations"
echo "(Released data layout)"
echo "============================================"
echo "Providers: ${PROVIDERS[*]}"
echo "Results:   $RESULTS_DIR"
echo ""

for provider in "${PROVIDERS[@]}"; do
    count=$(find "$RESULTS_DIR" -name "result_${provider}.json" 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "⚠️  No results for $provider — skipping"
        continue
    fi

    echo ""
    echo "========================================"
    echo "Provider: $provider ($count results)"
    echo "========================================"

    # Step 1: Tool accuracy (continuous scores)
    echo ""
    echo "--- [1/3] evaluate_tool_calls.py ---"
    python evaluate_tool_calls.py \
        --benchmark "$BENCHMARK" \
        --results-dir "$RESULTS_DIR" \
        --provider "$provider" \
        --output "${provider}_evaluation_report.json" \
        --use-llm || echo "⚠️  evaluate_tool_calls failed for $provider"

    # Step 2: Binary pass/fail rate
    echo ""
    echo "--- [2/3] evaluate_pass_rate.py ---"
    python evaluate_pass_rate.py \
        --benchmark "$BENCHMARK" \
        --results-dir "$RESULTS_DIR" \
        --provider "$provider" \
        --output "${provider}_pass_rate_report.json" \
        --use-llm || echo "⚠️  evaluate_pass_rate failed for $provider"

    # Step 3: Fine-grained latency analysis
    echo ""
    echo "--- [3/3] analyze_tool_latency.py ---"
    python analyze_tool_latency.py \
        --results-dir "$RESULTS_DIR" \
        --provider "$provider" \
        $SKIP_FLAG || echo "⚠️  analyze_tool_latency failed for $provider"

    echo ""
    echo "✅ $provider done"
done

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"
echo ""
echo "Generated reports:"
for provider in "${PROVIDERS[@]}"; do
    [ -f "${provider}_evaluation_report.json" ] && echo "  ${provider}_evaluation_report.json"
    [ -f "${provider}_pass_rate_report.json" ] && echo "  ${provider}_pass_rate_report.json"
    [ -f "${provider}_latency_report.json" ] && echo "  ${provider}_latency_report.json"
done
