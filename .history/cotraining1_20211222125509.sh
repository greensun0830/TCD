#!/bin/bash 
for((i=20;i<=30;i++));  
do   
python -u run_main1.py> /data1/home/qingyang/cotraining/dat/ml1m/3/$i.txt;  
done