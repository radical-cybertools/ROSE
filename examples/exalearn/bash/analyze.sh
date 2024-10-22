#!/bin/bash

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 1 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Orig: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 2 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "AL-p1: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 3 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Base-p1: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 4 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "AL-p2: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 5 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Base-p2: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 6 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "AL-p3: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg diff on test set" $i | head -n 7 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Base-p3: l2-diff: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

echo "##########################################"

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 1 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Orig: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 2 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "AL-p1: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 3 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Base-p1: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 4 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "AL-p2: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 5 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Base-p2: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 6 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "AL-p3: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg sigma^2 on test set" $i | head -n 7 | tail -n 1; done | awk '{sum += $7; sumsq += ($7)^2; n++} END {print "Base-p3: sigma^2: Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

echo "##########################################"

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 1 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "Orig: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 2 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "AL-p1: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 3 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "Base-p1: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 4 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "AL-p2: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 5 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "Base-p2: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 6 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "AL-p3: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'

for i in $(ls log*out); do grep "Avg class loss on test set" $i | head -n 7 | tail -n 1; done | awk '{sum += $8; sumsq += ($8)^2; n++} END {print "Base-p3: class loss Average:", sum/n; print "Standard Deviation:", sqrt(sumsq/n - (sum/n)^2)}'
