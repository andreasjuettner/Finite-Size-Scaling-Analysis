#!/bin/bash
conda activate intel-py37
export OMP_NUM_THREADS=1

# For logging
if [ ! -d "log" ]
then
    mkdir log
fi


for L in  128 96 64 48 32 16 8; do 
 for Bbar in 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59; do #0.56 0.59 0.57; do
  for g in 0.1 0.2 0.3 0.5 0.6; do 
   python3 Binderanalysis.py 2 $g $Bbar  $L > "log/log_2_"$g"_"$Bbar"_"$L  & sleep 2;  
  done 
 done
wait
done
wait
for L in  128 96 64 48 32 16 8; do 
 for Bbar in 0.42 0.43 0.44 0.45 0.46 0.47; do
  for g in 0.1 0.2 0.3 0.5 0.6; do 
   python3 Binderanalysis.py 4 $g $Bbar  $L > "log/log_4_"$g"_"$Bbar"_"$L  & sleep 2;  
  done 
 done
wait
done
