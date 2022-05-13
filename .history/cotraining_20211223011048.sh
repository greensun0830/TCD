#!/bin/bash 
for((i=1;i<=30;i++));  
do   
python -u run_main2.py > /amax/home/qingyang/cotraining/dat/ml100k/1/0/$i.txt;
done
