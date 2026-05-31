#!/bin/bash
# Run 4 parallel generators, each writing to separate file
# Then merge into train_v3.jsonl

echo "Starting 4 parallel generators..."

for i in 1 2 3 4; do
    nohup python3 -u /root/chimbisiai/scripts/generate_v2.py 500 \
        > /root/chimbisiai/gen_worker_${i}.log 2>&1 &
    echo "Worker $i: PID $!"
    sleep 2
done

echo "All workers started. Monitor with:"
echo "  wc -l /root/chimbisiai/data/train_v3.jsonl"
echo "  tail /root/chimbisiai/gen_worker_*.log"
