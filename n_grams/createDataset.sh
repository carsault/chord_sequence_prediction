#!/bin/sh
for alph in a0 a2 a5
do
	for var in 1 2 3 4 5
	do
		python3 code/create_data.py ../inputs/jazz_xlab/ --split train --seed $var --vocab $alph >datasets/train.seed$var.$alph.txt
		python3 code/create_data.py ../inputs/jazz_xlab/ --split valid --seed $var --vocab $alph >datasets/valid.seed$var.$alph.txt  
		python3 code/create_data.py ../inputs/jazz_xlab/ --split test --seed $var --vocab $alph >datasets/test.seed$var.$alph.txt 
	done
done