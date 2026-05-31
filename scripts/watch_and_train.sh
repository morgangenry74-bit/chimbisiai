#!/bin/bash
# Watch generation progress, start training when 2000 samples reached
DATA="/root/chimbisiai/data/train_v3.jsonl"
TARGET=2000

while true; do
    COUNT=$(wc -l < "$DATA" 2>/dev/null || echo 0)
    echo "[$(date)] Samples: $COUNT / $TARGET"
    
    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "[$(date)] Target reached! Stopping generator..."
        pkill -f generate_v2.py 2>/dev/null
        sleep 5
        echo "[$(date)] Starting v3 training..."
        cd /root/chimbisiai
        python3 -u scripts/train_v3.py 2>&1 | tee /root/chimbisiai/train_v3_log.txt
        echo "[$(date)] Training finished!"
        exit 0
    fi
    
    sleep 120
done
