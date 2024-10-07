@echo on
python main.py --max_epochs 60 --model efficientb0 --scheduler cyclic --opt SGD

python main.py --max_epochs 60 --model efficientb0 --scheduler cyclic2 --opt SGD

python main.py --max_epochs 60 --model efficientb0 --scheduler warmupcosine --opt SGD --e_patience 25

@echo off
