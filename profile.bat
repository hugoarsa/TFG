

nsys profile -w true -t cuda,nvtx,cublas -o efficientb0_report_100iters_base -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model efficientb0 --num_iter 100

nsys profile -w true -t cuda,nvtx,cublas -o res50_report_100iters_base -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model res50 --num_iter 100

nsys profile -w true -t cuda,nvtx,cublas -o dense121_report_100iters_base -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model dense121 --num_iter 100



nsys profile -w true -t cuda,nvtx,cublas -o batch_16 -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model efficientb0 --num_iter 50 --batch_size 16

nsys profile -w true -t cuda,nvtx,cublas -o batch_32 -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model efficientb0 --num_iter 50 --batch_size 32

nsys profile -w true -t cuda,nvtx,cublas -o batch_64 -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model efficientb0 --num_iter 50 --batch_size 64

nsys profile -w true -t cuda,nvtx,cublas -o batch_128 -f true --gpu-metrics-devices=all --capture-range=cudaProfilerApi --capture-range-end=stop python main.py --max_epochs 1 --model efficientb0 --num_iter 50 --batch_size 128