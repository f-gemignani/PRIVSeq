import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utility.constants as k

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows', 2000)



def risks2df(df_list):
    [df.sort_values(by=[k.USER_ID], inplace=True) for df in df_list]

    if not all([True if list((df_list[i])[k.USER_ID]) == list((df_list[0])[k.USER_ID]) else False for i in
                np.arange(1, len(df_list))]):
        raise ValueError("Attention! Uids included in path list don't match ")

    data = {'uids': list((df_list[0])[k.USER_ID])}
    for i in range(len(df_list)):
        attr = 'k' + str(i + 1)
        data[attr] = list((df_list[i])[PRIVACY_RISK])

    return pd.DataFrame(data)


def DrawAttackRiskDistribution(risks_df, title, cols=None):
    FONTSIZE = 15

    # For cumulative distribution figures
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.set_xlabel("risk", size=FONTSIZE)
    ax.set_ylabel('p(risk)', size=FONTSIZE)

    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.xticks(x, size=FONTSIZE - 5)
    plt.yticks(y, size=FONTSIZE - 5)

    ax.set_xlim([-0.02, 1.05])
    ax.set_ylim([-0.05, 1.05])

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyle_list = ['-.', 'dashed', ':', '-.', '--']

    # CUMULATE PART
    for i in range(len(risks_df.columns) - 1):
        _x = np.sort(risks_df['k' + str(i + 1)])
        _y = np.arange(1, len(_x) + 1) / len(_x)

        if cols == None:
            val = 'k=' + str(i + 1)
        else:
            val = cols[i]

        ax.step(_x, _y,
                color=color_list[i % len(color_list)],
                linewidth=1.5,
                linestyle=linestyle_list[i % len(linestyle_list)],
                label=val)

    # ax.set_title(title,fontsize = 12)
    plt.grid(alpha=0.3)
    plt.title(title, size=15)
    plt.legend(prop={'size': 10}, loc="lower center", bbox_to_anchor=(0.5, -0.25),

               ncol=4, fancybox=True)
    plt.tight_layout()
    plt.savefig("./imgs/" + title + ".png")
    plt.show()
