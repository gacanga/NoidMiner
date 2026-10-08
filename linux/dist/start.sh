#!/usr/bin/env bash
# NoidMiner - mines NOID with the settings of noidminer.conf.
# Edit noidminer.conf first: replace YOUR_NOID_ADDRESS with your NOID address.
# Restarts automatically after any exit (Ctrl+C twice to stop).
cd "$(dirname "$0")"
if grep -q "YOUR_NOID_ADDRESS" noidminer.conf; then
    echo "Edit noidminer.conf: replace YOUR_NOID_ADDRESS with your NOID payout address."
    echo "Modifiez noidminer.conf : remplacez YOUR_NOID_ADDRESS par votre adresse NOID."
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
trap 'echo; echo "stopped"; exit 0' INT TERM
while true; do
    ./noidminer --config noidminer.conf --log noidminer.log
    echo "NoidMiner exited (code $?), restarting in 10 s - Ctrl+C to stop."
    sleep 10
done
