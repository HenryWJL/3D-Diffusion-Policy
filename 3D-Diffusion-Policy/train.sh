torchrun --nproc_per_node=2 train.py --config-name=Freqpolicy.yaml \
                            task=robosuite_can \
                            task.dataset.zarr_path=/kaggle/working/3D-Diffusion-Policy/3D-Diffusion-Policy/data/robosuite_can.zarr \
                            hf_repo_id="HenryWJL/Freqpolicy" \
                            training.seed=0 \
                            training.num_epochs=3000 \
                            dataloader.batch_size=700 \
                            dataloader.num_workers=4 \
                            val_dataloader.num_workers=4 \
                            training.device="cuda" \
                            logging.mode=online \
                            checkpoint.save_ckpt=true \
                            training.rollout_every=4000 \
                            training.val_every=4000 \
                            training.sample_every=4000 \
                            training.checkpoint_every=600

torchrun --nproc_per_node=2 train.py --config-name=Freqpolicy.yaml \
                            task=robosuite_square \
                            task.dataset.zarr_path=/kaggle/working/3D-Diffusion-Policy/3D-Diffusion-Policy/data/robosuite_square.zarr \
                            hf_repo_id="HenryWJL/Freqpolicy" \
                            training.seed=2 \
                            training.num_epochs=3000 \
                            dataloader.batch_size=512 \
                            dataloader.num_workers=4 \
                            val_dataloader.num_workers=4 \
                            training.device="cuda" \
                            logging.mode=online \
                            checkpoint.save_ckpt=true \
                            training.rollout_every=4000 \
                            training.val_every=4000 \
                            training.sample_every=4000 \
                            training.checkpoint_every=600