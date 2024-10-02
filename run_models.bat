@echo on
:: Define the loss functions you want to test
set models=res50 dense121 efficientb3

:: Set the num_iter value
set max_epochs=30

:: Loop through each loss function
for %%M in (%models%) do (
    echo Running model: %%M for %max_epochs% epochs
    python main.py --model_name %%M --max_epochs %max_epochs%
)
@echo off