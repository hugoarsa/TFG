@echo on

python main.py --max_epochs 60 --model efficientb0

python main.py --max_epochs 30 --model efficientb0 --untrained

python main.py --max_epochs 60 --model res50

@echo off
