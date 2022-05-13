#!/bin/bash 
for((i=11;i<=40;i++));  
do   
python -u run_main.py > /data1/home/qingyang/cotraining/dat/ml1m/2/$i.txt;
done
