import matplotlib.pyplot as plt
import numpy as np


def plot_numbers(
    up_nums, down_nums, tick_labels=None, title=None, save_path=None, save_options=None
):
    total_nums = up_nums + down_nums
    L = total_nums.shape[0]
    indices = np.arange(L) - (L // 2 - 1)

    bar_width = ((6 - 1) / L) * 1.0 / 2

    plt.figure(figsize=(6, 4))

    plt.bar(indices, total_nums, width=2 * bar_width, alpha=0.2, color="green")
    plt.bar(indices - bar_width / 2, up_nums, bar_width, label="up")
    plt.bar(indices + bar_width / 2, down_nums, bar_width, label="down")

    if tick_labels is not None:
        plt.xticks(indices, tick_labels)
    else:
        plt.xticks(indices)
    if title is not None:
        plt.title(title)
    else:
        plt.title("Expected numbers")
    plt.ylabel("$\\langle n\\rangle$")
    plt.legend()
    if save_path is not None:
        if save_options is not None:
            plt.savefig(save_path, **save_options)
        else:
            plt.savefig(save_path)
    plt.show()


def generate_k_labels(
    n_momentum_points,
    k0,
):
    zero_index = -((n_momentum_points - 1) // 2)
    start_coeff = zero_index * k0
    coeffs = [start_coeff + i * k0 for i in range(n_momentum_points)]
    result_list = []
    for c in coeffs:
        if c == 0:
            # 계수가 0일 경우
            formatted_str = "$0$"
        elif c == 1:
            # 계수가 1일 경우
            formatted_str = "$k_r$"
        elif c == -1:
            # 계수가 -1일 경우
            formatted_str = "$-k_r$"
        else:
            # 그 외의 경우 (정수 및 실수)
            # 계수가 2.0, -3.0과 같이 정수이면 소수점(.0)을 제거합니다.
            if c == int(c):
                num_str = str(int(c))
            else:
                num_str = str(c)
            formatted_str = f"${num_str}k_r$"
        result_list.append(formatted_str)
    return result_list
