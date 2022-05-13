#!/bin/bash   
python -u run_main2.py --attack_size 0.01> /data1/home/qingyang/cotraining/dat/ml100k/1.txt;
python -u run_main2.py --attack_size 0.02> /data1/home/qingyang/cotraining/dat/ml100k/2.txt;
python -u run_main2.py --attack_size 0.04> /data1/home/qingyang/cotraining/dat/ml100k/4.txt;
python -u run_main2.py --attack_size 0.05> /data1/home/qingyang/cotraining/dat/ml100k/5.txt;