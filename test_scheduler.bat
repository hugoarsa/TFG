@echo on
:: Define the loss functions you want to test
set schedulers=cyclic cosine
:: Set the num_iter value
set num_iter=500
set max_epochs=30
set model=efficientb0

:: Loop through each loss function
for %%S in (%schedulers%) do (
    echo Running model with scheduler: %%S and num_iter: %num_iter%
    python main.py --scheduler %%S --num_iter %num_iter% --max_epochs %max_epochs% --model %model%
)

python main.py --scheduler cyclic --num_iter %num_iter% --max_epochs %max_epochs% --model %model% --lr 0.01

:: Define the loss functions you want to test
set optimizer=SGD SGD_Nesterov Adamax Adam AdamW RMSprop
:: Set the num_iter value
set num_iter=500
set max_epochs=30
set model=res18

:: Loop through each loss function
for %%O in (%optimizer%) do (
    echo Running model with optimizer: %%O and num_iter: %num_iter%
    python main.py --opt %%O --num_iter %num_iter% --max_epochs %max_epochs% --model %model%
)
@echo off