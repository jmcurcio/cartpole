import gymnasium as gym
import os
import datetime

def wrap_env_for_video(env, agent_name, episode_indices):
    """
    Wraps the environment with RecordVideo to record specified episodes.
    Returns the wrapped environment.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_folder = os.path.join('videos', agent_name, timestamp)
    os.makedirs(video_folder, exist_ok=True)

    def should_record(episode):
        return episode in episode_indices

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=should_record,
        name_prefix=f"{agent_name}",
        disable_logger=True
    )
    return env