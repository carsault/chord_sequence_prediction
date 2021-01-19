#!/bin/sh
for alph in a5
do
	for var in 5
	do
		python3 code/kenlm_eval.py models/seed$var.$alph.arpa ../datasets/"$alph"_1_"$var"label/test.pkl -n 8 --vocab $alph --beam 100 --output output
	done
done