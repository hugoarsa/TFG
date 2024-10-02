@echo on
:: Define the loss functions you want to test
set schedulers=plateau1 cyclic cosine warmupcosine
:: Set the num_iter value
set num_iter=500
set max_epochs=30
set model_name=efficientb0

:: Loop through each loss function
for %%S in (%schedulers%) do (
    echo Running model with scheduler: %%S and num_iter: %num_iter%
    python main.py --scheduler %%S --num_iter %num_iter% --max_epochs %max_epochs% --model_name %model_name%
)
@echo off