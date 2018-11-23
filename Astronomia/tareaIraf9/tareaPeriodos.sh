#!/bin/bash
if [[ $# -eq 0 ]] ; then
    echo 'Se debe pasar un Threshold como parámetro. ./tareaPeriodos.sh 3'
    exit 1
fi
rm -rf tareaFN
mkdir tareaFN
echo "Buscando candidatas con Threshold $1."
#Busca estrellas candidatas.
python -W ignore periodicas.py $1
FILE=candidatas.txt
cat $FILE | while read LINE; do
	name=$LINE
	wc -l $name >> cuentas
done
sort -n -r cuentas > cuentas2
rm cuentas
awk '{if ($1 > 19) print $2}' cuentas2 > cuentas3.txt
FILE=cuentas3.txt
touch tareaFN/names.txt
numCand=$(wc -l < $FILE)
echo "Se encontraron $numCand candidatas con 20 o más observaciones."
cat $FILE | while read LINE; do
	name=$LINE
	fnpeaks $name 0.001 100 0.00005 > "./tareaFN/$name.rta"
	echo $name >> "./tareaFN/names.txt"
done
python aIRAF.py
numCand=$(wc -l < tareaFN/candidatasAIRAF.txt)
echo "Se encontraron $numCand estrellas variables con periodo(s) mayor a 2 días."
cat tareaFN/periodoCandidatasAIRAF.txt
rm cuentas3.txt *.max
