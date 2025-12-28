policy=${1}
task=${2}

python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/1200.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/1800.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/2400.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/3000.pth