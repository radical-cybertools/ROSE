#!/bin/bash

#seed=42
#filename=/lus/eagle/projects/RECUP/twang/exalearn_stage2/experiment/seed_42/log_42_out

seed=41
filename=/lus/eagle/projects/RECUP/twang/exalearn_stage2/workflow/submit.sh.o1821932

echo "seed = ${seed}"

echo "##########################################"

printf "Orig:    l2-diff"
grep "Avg diff on test set" ${filename} | head -n 1 | tail -n 1

printf "AL-p1:   l2-diff"
grep "Avg diff on test set" ${filename} | head -n 2 | tail -n 1

printf "Base-p1: l2-diff"
grep "Avg diff on test set" ${filename} | head -n 3 | tail -n 1

printf "AL-p2:   l2-diff"
grep "Avg diff on test set" ${filename} | head -n 4 | tail -n 1

printf "Base-p2: l2-diff"
grep "Avg diff on test set" ${filename} | head -n 5 | tail -n 1

printf "AL-p3:   l2-diff"
grep "Avg diff on test set" ${filename} | head -n 6 | tail -n 1

printf "Base-p3: l2-diff"
grep "Avg diff on test set" ${filename} | head -n 7 | tail -n 1

echo "##########################################"

printf "Orig:    sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 1 | tail -n 1

printf "AL-p1:   sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 2 | tail -n 1

printf "Base-p1: sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 3 | tail -n 1

printf "AL-p2:   sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 4 | tail -n 1

printf "Base-p2: sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 5 | tail -n 1

printf "AL-p3:   sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 6 | tail -n 1

printf "Base-p3: sigma^2"
grep "Avg sigma^2 on test set" ${filename} | head -n 7 | tail -n 1

echo "##########################################"

printf "Orig:    class loss"
grep "Avg class loss on test set" ${filename} | head -n 1 | tail -n 1

printf "AL-p1:   class loss"
grep "Avg class loss on test set" ${filename} | head -n 2 | tail -n 1

printf "Base-p1: class loss"
grep "Avg class loss on test set" ${filename} | head -n 3 | tail -n 1

printf "AL-p2:   class loss"
grep "Avg class loss on test set" ${filename} | head -n 4 | tail -n 1

printf "Base-p2: class loss"
grep "Avg class loss on test set" ${filename} | head -n 5 | tail -n 1

printf "AL-p3:   class loss"
grep "Avg class loss on test set" ${filename} | head -n 6 | tail -n 1

printf "Base-p3: class loss"
grep "Avg class loss on test set" ${filename} | head -n 7 | tail -n 1
