#!/bin/bash 
for((i=1;i<=30;i++));  
do   
python -u run_main4.py > /data1/home/qingyang/cotraining/dat/filmtrust/1/4/$i.txt;  
done
