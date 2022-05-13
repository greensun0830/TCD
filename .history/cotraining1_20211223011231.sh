#!/bin/bash 
for((i=20;i<=30;i++));  
do   
python -u run_main3.py> /amax/home/qingyang/cotraining/dat/ml100k/1/1/$i.txt;  
done