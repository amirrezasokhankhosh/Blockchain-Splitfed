from data import *
import matplotlib.pyplot as plt


def plot_normal_9():
    plt.plot(range(1, len(split_learning) + 1), split_learning, label="Split Learning", color="C0")
    plt.plot(range(1, len(split_fed) + 1), split_fed, label="SplitFed Learning", color="C1")
    plt.plot(range(1, len(sharding_split_fed) + 1), sharding_split_fed, label="Sharding SplitFed", color="C2")
    plt.plot(range(1, len(bl_split_fed) + 1), bl_split_fed, label="Blockchain SplitFed", color="C3")

def plot_attack_9():
    plt.plot(range(1, len(attack_split_learning) + 1), attack_split_learning, label="Attacked Split Learning", color="C0", linestyle="dashed")
    plt.plot(range(1, len(attack_split_fed) + 1), attack_split_fed, label="Attacked SplitFed Learning", color="C1", linestyle="dashed")
    plt.plot(range(1, len(attack_sharding_split_fed) + 1), attack_sharding_split_fed, label="Attacked Sharding SplitFed", color="C2", linestyle="dashed")
    plt.plot(range(1, len(attack_bl_split_fed) + 1), attack_bl_split_fed, label="Attacked Blockchain Sharding SplitFed", color="C3", linestyle="dashed")

def plot_all_9():
    plot_normal_9()
    plot_attack_9()

def plot_normal_36():
    plt.plot(range(1, len(split_learning_36) + 1), split_learning_36, label="Split Learning", color="C0")
    plt.plot(range(1, len(split_fed_36) + 1), split_fed_36, label="SplitFed Learning", color="C1")
    plt.plot(range(1, len(sharding_split_fed_36) + 1), sharding_split_fed_36, label="Sharding SplitFed", color="C2")
    plt.plot(range(1, len(bl_split_fed_36) + 1), bl_split_fed_36, label="Blockchain SplitFed", color="C3")

def plot_attack_36():
    plt.plot(range(1, len(attack_split_learning_36) + 1), attack_split_learning_36, label="Attacked Split Learning", color="C0", linestyle="dashed")
    plt.plot(range(1, len(attack_split_fed_36) + 1), attack_split_fed_36, label="Attacked SplitFed Learning", color="C1", linestyle="dashed")
    plt.plot(range(1, len(attack_sharding_split_fed_36) + 1), attack_sharding_split_fed_36, label="Attacked Sharding SplitFed", color="C2", linestyle="dashed")
    plt.plot(range(1, len(attack_bl_split_fed_36) + 1), attack_bl_split_fed_36, label="Attacked Blockchain Sharding SplitFed", color="C3", linestyle="dashed")

def plot_all_36():
    plot_normal_36()
    plot_attack_36()


def plot_results(plot_algorithm, num_nodes, image_name, x_max):
    plt.figure(figsize=(10, 8))
    plot_algorithm()
    plt.grid(True, which='both', linestyle='--', linewidth=0.6)
    plt.title(f'Performance Comparison of Learning Approaches ({num_nodes} nodes)',
            fontsize=16, fontweight='bold')
    plt.xlabel('Communication Rounds', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks([1] + list(range(5, x_max+5, 5)), fontsize=12)
    plt.yticks(fontsize=12)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    plt.legend(title="Methods", fontsize=12, title_fontsize=13, loc='upper right')
    # plt.show()
    plt.savefig(f"./figures/{image_name}.png")


plot_results(plot_normal_9, num_nodes=9, image_name="noraml_9", x_max=60)
plot_results(plot_attack_9, num_nodes=9, image_name="attack_9", x_max=60)
plot_results(plot_all_9, num_nodes=9, image_name="all_9", x_max=60)
plot_results(plot_normal_36, num_nodes=36, image_name="noraml_36", x_max=30)
plot_results(plot_attack_36, num_nodes=36, image_name="attack_36", x_max=30)
plot_results(plot_all_36, num_nodes=36, image_name="all_36", x_max=30)
