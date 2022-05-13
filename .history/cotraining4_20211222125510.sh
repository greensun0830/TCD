#!/bin/bash   
python -u run_main.py --attack_size 0.01> /data1/home/qingyang/cotraining/dat/ml1m/1.txt;
python -u run_main.py --attack_size 0.02> /data1/home/qingyang/cotraining/dat/ml1m/2.txt;
python -u run_main.py --attack_size 0.04> /data1/home/qingyang/cotraining/dat/ml1m/4.txt;
python -u run_main.py --attack_size 0.05> /data1/home/qingyang/cotraining/dat/ml1m/5.txt;