@echo on
:: Define the loss functions you want to test
set loss_functions=bce bce_w focal asymmetric asymmetric_avg asl1 asl2 asl3

:: Set the num_iter value
set num_iter=250
set max_epochs=25

:: Loop through each loss function
for %%L in (%loss_functions%) do (
    echo Running model with loss function: %%L and num_iter: %num_iter%
    python main.py --loss %%L --num_iter %num_iter% --max_epochs %max_epochs%
)
@echo off