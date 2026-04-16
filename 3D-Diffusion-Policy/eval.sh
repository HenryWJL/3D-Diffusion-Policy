policy=${1}
task=${2}
device=${3}

python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D600_seed%3D0.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D1200_seed%3D0.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D1800_seed%3D0.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D2400_seed%3D0.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D3000_seed%3D0.pth

python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D600_seed%3D1.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D1200_seed%3D1.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D1800_seed%3D1.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D2400_seed%3D1.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D3000_seed%3D1.pth

python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D600_seed%3D2.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D1200_seed%3D2.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D1800_seed%3D2.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D2400_seed%3D2.pth
python 3D-Diffusion-Policy/eval.py --config-name=${policy}.yaml task=${task} training.device=${device} ckpt_paths=checkpoints/epoch%3D3000_seed%3D2.pth