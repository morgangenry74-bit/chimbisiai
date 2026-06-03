#!/bin/bash
cd /root/chimbisiai
python3 -u scripts/generate_v2_async.py 2000 >> logs/generate_async.log 2>&1
