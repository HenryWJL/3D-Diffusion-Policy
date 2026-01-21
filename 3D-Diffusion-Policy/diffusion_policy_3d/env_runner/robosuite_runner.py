import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.env import RobosuiteEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from diffusion_policy_3d.common.svgd import ActionSampler, svgd_gradient, svgd_update
from termcolor import cprint


class RobosuiteRunner(BaseRunner):

    def __init__(
        self,
        output_dir,
        shape_meta,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        abs_action=True,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        task_name=None,
        bounding_boxes=dict(),
    ):
        super().__init__(output_dir)
        self.shape_meta = shape_meta
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    env=RobosuiteEnv(
                        env_name=task_name,
                        robots="Panda",
                        camera_names=list(bounding_boxes.keys()),
                        bounding_boxes=bounding_boxes,
                        delta_action=not abs_action,
                        render_image_size=(render_size, render_size)
                    ),
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        test_start_seed = 10000

        all_goal_achieved = []
        all_success_rates = []
        videos = []
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Robosuite {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            self.env.env.env.seed(test_start_seed + episode_idx)
            obs = self.env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = {key: obs[key] for key in self.shape_meta['obs'].keys() if not key.endswith('pc_mask')}
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(
                                          device=device))
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = self.env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                num_goal_achieved += np.sum(info['is_success'])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(np.sum(info['is_success']))
            all_goal_achieved.append(num_goal_achieved)
            videos.append(self.env.env.get_video())

        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # log_data[f'sim_video_eval'] = videos_wandb
        
        # Save videos
        import imageio
        videos = np.transpose(np.concatenate(videos), (0, 2, 3, 1))  # -> (T, H, W, C)
        imageio.mimwrite("rollout.mp4", videos, fps=30, codec='libx264')

        # clear out video buffer
        _ = self.env.reset()
        # # clear memory
        # videos = None
        # del env

        return log_data



# class RobosuiteRunner(BaseRunner):

#     def __init__(
#         self,
#         output_dir,
#         shape_meta,
#         eval_episodes=20,
#         max_steps=200,
#         n_obs_steps=8,
#         n_action_steps=8,
#         abs_action=True,
#         fps=10,
#         crf=22,
#         render_size=84,
#         tqdm_interval_sec=5.0,
#         task_name=None,
#         bounding_boxes=dict(),
#     ):
#         super().__init__(output_dir)
#         self.shape_meta = shape_meta
#         self.task_name = task_name

#         steps_per_render = max(10 // fps, 1)

#         def env_fn():
#             return MultiStepWrapper(
#                 SimpleVideoRecordingWrapper(
#                     env=RobosuiteEnv(
#                         env_name=task_name,
#                         robots="Panda",
#                         camera_names=list(bounding_boxes.keys()),
#                         bounding_boxes=bounding_boxes,
#                         delta_action=not abs_action,
#                         render_image_size=(render_size, render_size)
#                     ),
#                     steps_per_render=steps_per_render
#                 ),
#                 n_obs_steps=n_obs_steps,
#                 n_action_steps=n_action_steps,
#                 max_episode_steps=max_steps,
#                 reward_agg_method='sum',
#             )

#         self.eval_episodes = eval_episodes
#         self.env = env_fn()

#         self.fps = fps
#         self.crf = crf
#         self.n_obs_steps = n_obs_steps
#         self.n_action_steps = n_action_steps
#         self.max_steps = max_steps
#         self.tqdm_interval_sec = tqdm_interval_sec

#         self.logger_util_test = logger_util.LargestKRecorder(K=3)
#         self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

#     def run(self, policy: BasePolicy):
#         device = policy.device
#         test_start_seed = 10000

#         all_goal_achieved = []
#         all_success_rates = []
#         videos = []

#         action_sampler = ActionSampler(
#             policy,
#             16,
#             self.n_obs_steps,
#             self.n_action_steps,
#             self.shape_meta['action']['shape'][0],
#             self.max_steps,
#             step_size=1e-2
#         )
        
#         for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Robosuite {self.task_name} Pointcloud Env",
#                                      leave=False, mininterval=self.tqdm_interval_sec):
                
#             # start rollout
#             self.env.env.env.seed(test_start_seed + episode_idx)
#             obs = self.env.reset()
#             action_sampler.reset()

#             done = False
#             num_goal_achieved = 0
#             actual_step_count = 0

#             while not done:
#                 # create obs dict
#                 np_obs_dict = {key: obs[key] for key in self.shape_meta['obs'].keys() if not key.endswith('pc_mask')}
#                 # device transfer
#                 obs_dict = dict_apply(np_obs_dict,
#                                       lambda x: torch.from_numpy(x).unsqueeze(0).to(
#                                           device=device))
#                 action = action_sampler.update(obs_dict)
#                 action = action.detach().to('cpu').unsqueeze(0).numpy()
#                 # # step env
#                 obs, reward, done, info = self.env.step(action)
#                 # all_goal_achieved.append(info['goal_achieved'])
#                 num_goal_achieved += np.sum(info['is_success'])
#                 done = np.all(done)
#                 actual_step_count += 1

#             all_success_rates.append(np.sum(info['is_success']))
#             all_goal_achieved.append(num_goal_achieved)
#             videos.append(self.env.env.get_video())

#         # log
#         log_data = dict()
        

#         log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
#         log_data['mean_success_rates'] = np.mean(all_success_rates)

#         log_data['test_mean_score'] = np.mean(all_success_rates)

#         cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

#         self.logger_util_test.record(np.mean(all_success_rates))
#         self.logger_util_test10.record(np.mean(all_success_rates))
#         log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
#         log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

#         # videos = env.env.get_video()
#         # if len(videos.shape) == 5:
#         #     videos = videos[:, 0]  # select first frame
#         # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
#         # log_data[f'sim_video_eval'] = videos_wandb
        
#         # Save videos
#         import imageio
#         videos = np.transpose(np.concatenate(videos), (0, 2, 3, 1))  # -> (T, H, W, C)
#         imageio.mimwrite("rollout.mp4", videos, fps=30, codec='libx264')

#         # clear out video buffer
#         _ = self.env.reset()
#         # # clear memory
#         # videos = None
#         # del env

#         return log_data



# def receding_horizon_select(
#     current_batch_actions, 
#     prev_action_seq, 
#     lookahead_steps=8,
#     low_pass_cutoff=None,
#     decay_rate=1.0
# ):
#     """
#     Receding horizon spectral selection.
    
#     Args:
#         current_batch_actions: (B, L, D) np.ndarray - Batch of candidates at time t
#         prev_action_seq: (L, D) np.ndarray - The chosen sequence from time t-1
#         lookahead_steps: int - Window size for consistency check
#         low_pass_cutoff: int or None - Index to stop summing frequencies
#         decay_rate: float - Strength of low-frequency bias (higher = smoother)
#     """
#     B, L, D = current_batch_actions.shape
    
#     # --- 1. The Receding Horizon Slice ---
#     safe_lookahead = min(lookahead_steps, L - 1)
    
#     # Slice the intersection
#     # Target (Previous plan): Indices [1 : 1+lookahead]
#     target_slice = prev_action_seq[1 : 1 + safe_lookahead]       # Shape (Lookahead, D)
    
#     # Candidates (Current plan): Indices [0 : lookahead]
#     candidate_slice = current_batch_actions[:, :safe_lookahead, :] # Shape (B, Lookahead, D)

#     # --- 2. FFT (Norm="ortho") ---
#     # Numpy uses 'axis' instead of 'dim'.
#     # norm="ortho" ensures energy conservation (Parseval's theorem).
    
#     # FFT over time axis (axis=0 for target, axis=1 for batch candidates)
#     fft_target = np.fft.rfft(target_slice, axis=0, norm="ortho")         # Shape (Freqs, D)
#     fft_candidates = np.fft.rfft(candidate_slice, axis=1, norm="ortho")  # Shape (B, Freqs, D)

#     # --- 3. Spectral Distance Calculation ---
#     # We broaden fft_target to (1, Freqs, D) to broadcast against (B, Freqs, D)
#     diff = fft_candidates - fft_target[None, :, :]
    
#     # Squared Magnitude Difference (Energy of error)
#     # np.abs() handles complex numbers correctly
#     dist_sq = np.abs(diff) ** 2  # Shape (B, Freqs, D)

#     # --- 4. Frequency Filtering & Weighting ---
    
#     # A. Hard Cutoff (Optional)
#     if low_pass_cutoff is not None:
#         dist_sq = dist_sq[:, :low_pass_cutoff, :]
    
#     # B. Soft Weighting
#     num_freqs = dist_sq.shape[1]
#     freq_indices = np.arange(num_freqs)
    
#     # Exponential decay weights
#     weights = np.exp(-decay_rate * freq_indices)
    
#     # Normalize weights so they sum to 1
#     weights = weights / np.sum(weights)
    
#     # Reshape for broadcasting: (1, Freqs, 1)
#     # This aligns weights with the Frequency dimension of dist_sq
#     weights = weights[None, :, None]

#     # --- 5. Scoring ---
#     # Apply weights
#     weighted_dist = dist_sq * weights
    
#     # Sum over Frequencies (axis 1) and Action Dimensions (axis 2)
#     scores = np.sum(weighted_dist, axis=(1, 2)) # Result shape (B,)

#     # --- 6. Selection ---
#     best_idx = np.argmin(scores)
#     best_action = current_batch_actions[best_idx]
    
#     return best_action, best_idx



# class RobosuiteRunner(BaseRunner):

#     def __init__(
#         self,
#         output_dir,
#         shape_meta,
#         eval_episodes=20,
#         max_steps=200,
#         n_obs_steps=8,
#         n_action_steps=8,
#         abs_action=True,
#         fps=10,
#         crf=22,
#         render_size=84,
#         tqdm_interval_sec=5.0,
#         task_name=None,
#         bounding_boxes=dict(),
#     ):
#         super().__init__(output_dir)
#         self.shape_meta = shape_meta
#         self.task_name = task_name

#         steps_per_render = max(10 // fps, 1)

#         def env_fn():
#             return MultiStepWrapper(
#                 SimpleVideoRecordingWrapper(
#                     env=RobosuiteEnv(
#                         env_name=task_name,
#                         robots="Panda",
#                         camera_names=list(bounding_boxes.keys()),
#                         bounding_boxes=bounding_boxes,
#                         delta_action=not abs_action,
#                         render_image_size=(render_size, render_size)
#                     ),
#                     steps_per_render=steps_per_render
#                 ),
#                 n_obs_steps=n_obs_steps,
#                 n_action_steps=n_action_steps,
#                 max_episode_steps=max_steps,
#                 reward_agg_method='sum',
#             )

#         self.eval_episodes = eval_episodes
#         self.env = env_fn()

#         self.fps = fps
#         self.crf = crf
#         self.n_obs_steps = n_obs_steps
#         self.n_action_steps = n_action_steps
#         self.max_steps = max_steps
#         self.tqdm_interval_sec = tqdm_interval_sec

#         self.logger_util_test = logger_util.LargestKRecorder(K=3)
#         self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

#     def run(self, policy: BasePolicy):
#         policy.eval()
#         device = policy.device
#         test_start_seed = 10000

#         all_goal_achieved = []
#         all_success_rates = []
#         videos = []
#         K = 5
        
#         for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Robosuite {self.task_name} Pointcloud Env",
#                                      leave=False, mininterval=self.tqdm_interval_sec):
                
#             # start rollout
#             self.env.env.env.seed(test_start_seed + episode_idx)
#             obs = self.env.reset()

#             done = False
#             num_goal_achieved = 0
#             actual_step_count = 0
#             prev_actions = None

#             while not done:
#                 # create obs dict
#                 np_obs_dict = {key: obs[key] for key in self.shape_meta['obs'].keys() if not key.endswith('pc_mask')}
#                 if prev_actions is None:  # First prediction
#                     # device transfer
#                     obs_dict = dict_apply(np_obs_dict,
#                                         lambda x: torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(
#                                             device=device))
#                     # Predict
#                     with torch.no_grad():
#                         result = policy.predict_action(obs_dict)
#                         normalized_actions = policy.normalizer['action'].normalize(result['action']).squeeze(0).detach().to('cpu').numpy()
#                         actions = result['action'].squeeze(0).detach().to('cpu').numpy()
#                         prev_actions = normalized_actions
#                 else:
#                     # device transfer
#                     obs_dict = dict_apply(np_obs_dict,
#                                         lambda x: torch.from_numpy(x.astype(np.float32)).unsqueeze(0).expand(K, *x.shape).to(
#                                             device=device))
#                     # Predict
#                     with torch.no_grad():
#                         result = policy.predict_action(obs_dict)
#                         normalized_actions = policy.normalizer['action'].normalize(result['action']).detach().to('cpu').numpy()
#                         actions = result['action'].detach().to('cpu').numpy()
#                         best_normalized_actions, best_idx = receding_horizon_select(normalized_actions, prev_actions)
#                         actions = actions[best_idx]
#                         prev_actions = best_normalized_actions
                    
#                 # with torch.no_grad():
#                 #     result = policy.predict_action(obs_dict)
#                 #     action_preds = result['action_pred']
#                     # actions = result['action'].float()
#                     # score_preds = policy.compute_score(action_preds, obs_dict).float()
#                     # start = self.n_obs_steps - 1
#                     # end = start + self.n_action_steps
#                     # scores = score_preds[:, start: end]
#                     # grads = svgd_gradient(actions, scores)
#                     # actions = actions + 0.01 * grads
#                     # action_preds = svgd_update(action_preds, obs_dict, policy)
#                     # start = self.n_obs_steps - 1
#                     # end = start + self.n_action_steps
#                     # actions = action_preds[:, start: end]
#                     # action = actions.mean(dim=0)
#                     # action = action.detach().to('cpu').numpy()
#                 # # step env
#                 obs, reward, done, info = self.env.step(actions[:1])

#                 # np_obs_dict = {key: obs[key] for key in self.shape_meta['obs'].keys() if not key.endswith('pc_mask')}
#                 # # device transfer
#                 # obs_dict = dict_apply(np_obs_dict,
#                 #                     lambda x: torch.from_numpy(x).unsqueeze(0).to(
#                 #                         device=device))
#                 # for i in range(1, actions.shape[0]):
#                 #     old_score = scores[i]
#                 #     action = actions[i].unsqueeze(0).detach().to('cpu').numpy()
#                 #     padded_action_preds = torch.zeros_like(action_preds)
#                 #     padded_action_preds[:, :-i] = action_preds[:, i:]
#                 #     with torch.no_grad():
#                 #         new_score_preds = policy.compute_score(padded_action_preds, obs_dict, noise, timesteps)
#                 #     start = self.n_obs_steps - 1
#                 #     end = start + self.n_action_steps
#                 #     new_scores = new_score_preds[:, start: end].squeeze(0)
#                 #     new_score = new_scores[0]
#                 #     # scores[i:] = new_scores[:-i]
#                 #     div = (torch.sqrt(((new_score - old_score) ** 2).sum())) / self.shape_meta['action']['shape'][0]
#                 #     # div = ((new_score - old_score) ** 2).mean() / self.shape_meta['action']['shape'][0] ** 2
#                 #     if div < 100.0:
#                 #         obs, reward, done, info = self.env.step(action)
#                 #         np_obs_dict = {key: obs[key] for key in self.shape_meta['obs'].keys() if not key.endswith('pc_mask')}
#                 #         # device transfer
#                 #         obs_dict = dict_apply(np_obs_dict,
#                 #                             lambda x: torch.from_numpy(x).unsqueeze(0).to(
#                 #                                 device=device))
#                 #     else:
#                 #         break

#                 # obs, reward, done, info = self.env.step(action)
#                 num_goal_achieved += np.sum(info['is_success'])
#                 done = np.all(done)
#                 actual_step_count += 1

#             all_success_rates.append(np.sum(info['is_success']))
#             all_goal_achieved.append(num_goal_achieved)
#             videos.append(self.env.env.get_video())

#         # log
#         log_data = dict()
        

#         log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
#         log_data['mean_success_rates'] = np.mean(all_success_rates)

#         log_data['test_mean_score'] = np.mean(all_success_rates)

#         cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

#         self.logger_util_test.record(np.mean(all_success_rates))
#         self.logger_util_test10.record(np.mean(all_success_rates))
#         log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
#         log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

#         # videos = env.env.get_video()
#         # if len(videos.shape) == 5:
#         #     videos = videos[:, 0]  # select first frame
#         # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
#         # log_data[f'sim_video_eval'] = videos_wandb
        
#         # Save videos
#         import imageio
#         videos = np.transpose(np.concatenate(videos), (0, 2, 3, 1))  # -> (T, H, W, C)
#         imageio.mimwrite("rollout.mp4", videos, fps=30, codec='libx264')

#         # clear out video buffer
#         _ = self.env.reset()
#         # # clear memory
#         # videos = None
#         # del env

#         return log_data


