import matplotlib.pyplot as plt

def plot_rewards(rewards, show=True):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards Over Time")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    plt.close() 