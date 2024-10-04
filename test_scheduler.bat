@echo on
:: Define the loss functions you want to test
set img_sizes=256 224 128 64
:: Set the num_iter value
set num_iter=500
set max_epochs=20
set model=efficientb0

:: Loop through each loss function
for %%I in (%img_sizes%) do (
    echo Running model with image_size: %%I and num_iter: %num_iter%
    python main.py --img_size %%I --num_iter %num_iter% --max_epochs %max_epochs% --model %model%
)

python main.py --num_iter %num_iter% --max_epochs 30 --model %model% --scheduler warmupcosine

python main.py --num_iter %num_iter% --max_epochs 30 --model %model% --scheduler cosine --lr 0.005

python main.py --max_epochs 30 --model %model% --pretrained False

