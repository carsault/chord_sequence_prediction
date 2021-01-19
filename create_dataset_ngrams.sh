#!/bin/sh
for alph in a0 a2 a5
do
	for var in 1 2 3 4 5
	do
		python3 buildDataset.py  --random_state $var --alpha $alph --label True
	done
done