#!/bin/bash 
for((i=1;i<=30;i++));  
do   
python -u run_main3.py > /data1/home/qingyang/cotraining/dat/ml100k/1/1/$i.txt;  
done