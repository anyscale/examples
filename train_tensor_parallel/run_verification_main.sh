#!/bin/bash
# Verification script using main training scripts
# This script runs all three implementations with shared initial weights
# and generates a comparison plot.

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
NUM_LAYERS="${NUM_LAYERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DEBUG_STEPS="${DEBUG_STEPS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# Output directories
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/loss_curves_verify}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/tmp/shared_weights_verify}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/ray_checkpoints_verify}"

# Create directories
mkdir -p "$OUTPUT_DIR" "$WEIGHTS_DIR"

echo "============================================================"
echo "Verification Using Main Training Scripts"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Layers: $NUM_LAYERS"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LENGTH"
echo "Learning rate: $LEARNING_RATE"
echo "Debug steps: $DEBUG_STEPS"
echo "Output dir: $OUTPUT_DIR"
echo "============================================================"

# Step 1: Run DDP (baseline) and save initial weights
echo ""
echo "[Step 1/4] Running DDP (baseline) and saving initial weights..."
echo "------------------------------------------------------------"
python train_ddp.py \
    --model_name "$MODEL_NAME" \
    --num_workers 2 \
    --batch_size "$BATCH_SIZE" \
    --seq_length "$SEQ_LENGTH" \
    --num_layers "$NUM_LAYERS" \
    --num_epochs 1 \
    --learning_rate "$LEARNING_RATE" \
    --debug_steps "$DEBUG_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --loss_output_dir "$OUTPUT_DIR" \
    --storage_path "$CHECKPOINT_DIR" \
    --init_weights_path "$WEIGHTS_DIR/init_weights.pt" \
    --save_init_weights

echo ""
echo "DDP completed. Initial weights saved to $WEIGHTS_DIR/init_weights.pt"

# Step 2: Run DeepSpeed AutoTP with same weights
echo ""
echo "[Step 2/4] Running DeepSpeed AutoTP with shared weights..."
echo "------------------------------------------------------------"
python train_deepspeed.py \
    --model_name "$MODEL_NAME" \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --batch_size "$BATCH_SIZE" \
    --seq_length "$SEQ_LENGTH" \
    --num_layers "$NUM_LAYERS" \
    --num_epochs 1 \
    --learning_rate "$LEARNING_RATE" \
    --debug_steps "$DEBUG_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --loss_output_dir "$OUTPUT_DIR" \
    --storage_path "$CHECKPOINT_DIR" \
    --init_weights_path "$WEIGHTS_DIR/init_weights.pt"

echo ""
echo "DeepSpeed AutoTP completed."

# Step 3: Run FSDP+DTensor with same weights
echo ""
echo "[Step 3/4] Running FSDP+DTensor with shared weights..."
echo "------------------------------------------------------------"
python train_fsdp.py \
    --model_name "$MODEL_NAME" \
    --tp_size 2 \
    --dp_size 2 \
    --num_workers 4 \
    --batch_size "$BATCH_SIZE" \
    --seq_length "$SEQ_LENGTH" \
    --num_layers "$NUM_LAYERS" \
    --num_epochs 1 \
    --learning_rate "$LEARNING_RATE" \
    --debug_steps "$DEBUG_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --loss_output_dir "$OUTPUT_DIR" \
    --storage_path "$CHECKPOINT_DIR" \
    --init_weights_path "$WEIGHTS_DIR/init_weights.pt" \
    --autocast

echo ""
echo "FSDP+DTensor completed."

# Step 4: Plot and compare loss curves
echo ""
echo "[Step 4/4] Plotting loss curves..."
echo "------------------------------------------------------------"
python plot_loss_curves.py \
    --input_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/loss_curves.png"

echo ""
echo "============================================================"
echo "Verification Complete!"
echo "============================================================"
echo "Loss history files:"
ls -la "$OUTPUT_DIR"/*.json
echo ""
echo "Plot saved to: $OUTPUT_DIR/loss_curves.png"
echo "============================================================"
