#!/bin/bash
./sigcol 966.WFI.dat 3 2 | grep ID > rta.txt
for filen in *.dat; do
	./sigcol $filen 3 2 | grep -v ID >> rta.txt
done
