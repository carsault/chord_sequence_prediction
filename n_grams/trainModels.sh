#!/bin/sh
for alph in a0 a2 a5
do
	for var in 1 2 3 4 5
	do
		cat datasets/train.seed$var.$alph.txt  datasets/valid.seed$var.$alph.txt | testKenLMV2/kenlm/build/bin/lmplz -o 9 --limit_vocab_file datasets/vocab.$alph.txt --discount_fallback>models/seed$var.$alph.arpa 
	done
done