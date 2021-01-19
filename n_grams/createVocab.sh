#!/bin/sh
for alph in a0 a2 a5
do
	python3 code/create_data.py ../inputs/jazz_xlab/ --vocab $alph --vocab_only>datasets/vocab.$alph.txt
done