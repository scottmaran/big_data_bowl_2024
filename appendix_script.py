import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_plot_1():

    xt_1 = 0.8
    xt_2 = 0.2

    def per_timestep_toe(xt, num_samples=1000):
        x = np.random.binomial(n=1, p=xt, size=num_samples)
        toe = (1 - xt)*x + (0 - xt)*np.abs(x-1)
        return toe

    toe_1 = per_timestep_toe(xt_1, num_samples=1000000)
    toe_2 = per_timestep_toe(xt_2, num_samples=1000000)

    fig, ax = plt.subplots(1,2)
    pd.Series(toe_1).hist(ax=ax[0])
    pd.Series(toe_2).hist(ax=ax[1])
    ax[0].set_title("Player 1 TOE distribution")
    ax[1].set_title("Player 2 TOE distribution")
    plt.show()

    print(f"P1 dist = mean={np.round(np.mean(toe_1),3)}, var={np.round(np.var(toe_1),3)}")
    print(f"P2 dist = mean={np.round(np.mean(toe_2), 3)} var={np.round(np.var(toe_2),3)}")


def gen_plot_2():
    predicted_xt_1 = 0.8
    true_xt_1 = 0.9
    predicted_xt_2 = 0.2
    true_xt_2 = 0.3

    def per_timestep_diff_abilities_toe(true_xt, predicted_xt, num_samples=1000):
        x = np.random.binomial(n=1, p=true_xt, size=num_samples)
        toe = (1 - predicted_xt)*x + (0 - predicted_xt)*np.abs(x-1)
        return toe

    varying_talent_toe_1 = per_timestep_diff_abilities_toe(true_xt_1, predicted_xt_1, num_samples=1000000)
    varying_talent_toe_2 = per_timestep_diff_abilities_toe(true_xt_2, predicted_xt_2, num_samples=1000000)

    fig, ax = plt.subplots(1,2)
    pd.Series(varying_talent_toe_1).hist(ax=ax[0])
    pd.Series(varying_talent_toe_2).hist(ax=ax[1])
    ax[0].set_title("Player 1 TOE distribution")
    ax[1].set_title("Player 2 TOE distribution")
    plt.show()

    print(f"P1 dist = mean={np.round(np.mean(varying_talent_toe_1),3)}, var={np.round(np.var(varying_talent_toe_1),3)}")
    print(f"P2 dist = mean={np.round(np.mean(varying_talent_toe_2), 3)} var={np.round(np.var(varying_talent_toe_2),3)}")