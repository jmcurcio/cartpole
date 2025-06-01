import gymnasium as gym
import os
import datetime

def wrap_env_for_video(env, episode_indices, current_episode, agent_name):
    """
    Wraps the environment with RecordVideo if the current episode is in episode_indices.
    Returns the (possibly wrapped) environment and a boolean indicating if recording is active.
    """
    video_folder = os.path.join('videos', agent_name)
    if current_episode in episode_indices:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: True,
            name_prefix=f'{agent_name}_ep{current_episode+1}_{timestamp}'
        )
        return env, True
    return env, False