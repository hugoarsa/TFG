@echo on

python main.py --max_epochs 60 --model efficientb0

python main.py --max_epochs 60 --model densenet121

python main.py --max_epochs 60 --model res50

python main.py --max_epochs 60 --model efficientb0 --scheduler warmupcosine

@echo off
