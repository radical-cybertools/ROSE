#!/bin/bash

grep -E "epoch [0-9]+ takes" log_1_2048_1_2_NUMA | tail -n +2 | awk '{sum += $4; count++} END {print sum/count}'
