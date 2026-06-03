#!/bin/bash
pkill -f "dashboard_fix.py" 2>/dev/null
sleep 1
cd /root/chimbisiai
nohup python3 dashboard_fix.py > /tmp/dash.log 2>&1 &
echo $!
