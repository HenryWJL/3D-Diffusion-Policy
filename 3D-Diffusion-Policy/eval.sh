policy=${1}
task=${2}

# python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/epoch%3D200_seed%3D0.pth
# python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="cpu" ckpt_paths=checkpoints/epoch%3D400_seed%3D0.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="mps" ckpt_paths=checkpoints/epoch%3D600_seed%3D2.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="mps" ckpt_paths=checkpoints/epoch%3D1200_seed%3D2.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device="mps" ckpt_paths=checkpoints/epoch%3D1800_seed%3D2.pth