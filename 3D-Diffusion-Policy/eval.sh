policy=${1}
task=${2}

python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/200.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/400.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/600.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/800.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/1000.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/1200.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/1400.pth