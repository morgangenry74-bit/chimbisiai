#!/usr/bin/env python3
"""
CHIMBISIAI Training Monitor
Checks dataset generation progress and auto-starts training when ready.
Run via cron every 2 hours.
"""
import subprocess
import json
import sys

GPU_HOST = "root@vm-6691.user-project-2874.cloud.intcld.ru"
DATASET_FILE = "/root/chimbisiai/data/train_v3.jsonl"
MIN_SAMPLES = 500  # minimum to start training
TARGET_SAMPLES = 2000
TRAIN_SCRIPT = "/root/chimbisiai/scripts/train_v2.py"
LOG_FILE = "/root/chimbisiai/train_v2_log.txt"


def ssh_cmd(cmd, timeout=30):
    """Run command on GPU server"""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", GPU_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip(), result.returncode


def check_status():
    """Check generation and training status"""
    # Check if generation is still running
    gen_running, _ = ssh_cmd("ps aux | grep generate_v2 | grep -v grep | wc -l")
    gen_running = int(gen_running or "0")

    # Check dataset size
    count, _ = ssh_cmd(f"wc -l {DATASET_FILE} 2>/dev/null | cut -d' ' -f1")
    count = int(count or "0")

    # Check if training is already running
    train_running, _ = ssh_cmd("ps aux | grep train_v2 | grep -v grep | wc -l")
    train_running = int(train_running or "0")

    # Check if trained model exists
    model_exists, rc = ssh_cmd("ls /root/chimbisiai/output/chimbisiai-v2-merged 2>/dev/null | wc -l")
    model_exists = int(model_exists or "0") > 0

    return {
        "gen_running": gen_running > 0,
        "dataset_count": count,
        "train_running": train_running > 0,
        "model_exists": model_exists,
    }


def start_training():
    """Start training on GPU server"""
    print("Starting training...")
    ssh_cmd(
        f"nohup python3 -u {TRAIN_SCRIPT} > {LOG_FILE} 2>&1 &",
        timeout=10
    )


def main():
    status = check_status()
    print(f"Status: {json.dumps(status, indent=2)}")

    if status["model_exists"]:
        print("✅ Model v2 already trained! Nothing to do.")
        return "done"

    if status["train_running"]:
        print("⏳ Training in progress...")
        return "training"

    if status["gen_running"]:
        print(f"⏳ Generation in progress: {status['dataset_count']}/{TARGET_SAMPLES} samples")
        if status["dataset_count"] >= MIN_SAMPLES:
            print(f"  → Enough samples ({status['dataset_count']} >= {MIN_SAMPLES}), could start training")
            # Don't start yet - wait for generation to finish for best quality
            if status["dataset_count"] >= TARGET_SAMPLES * 0.8:
                print("  → 80%+ done, starting training!")
                start_training()
                return "started_training"
        return "generating"

    # Generation finished
    if status["dataset_count"] >= MIN_SAMPLES:
        print(f"✅ Generation complete: {status['dataset_count']} samples. Starting training!")
        start_training()
        return "started_training"
    else:
        print(f"❌ Generation stopped with only {status['dataset_count']} samples (need {MIN_SAMPLES})")
        return "error"


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
