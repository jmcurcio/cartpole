import gymnasium as gym
import os

def wrap_env_for_video(env, video_dir, episode_indices, current_episode, agent_name):
    """
    Wraps the environment with RecordVideo if the current episode is in episode_indices.
    Returns the (possibly wrapped) environment and a boolean indicating if recording is active.
    """
    if current_episode in episode_indices:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_dir=video_dir,
            episode_trigger=lambda ep: True,  # record this episode only
            name_prefix=f"{agent_name}_ep{current_episode+1}"
        )
        return env, True
    return env, False 