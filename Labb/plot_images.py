import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def display_images(data, get_labels, path, title, nrows=2, ncols=5, figsize=(15, 8)):
    """
    :param data: dataframe, list or np.array
    :param get_labels:
    :param path: str
    :param title: str
    :param nrows: int
    :param ncols: int
    :param figsize: tuple int
    :return: None
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if path:
        for pict, ax in zip(data, axes.flatten()):
            img = plt.imread(f"{path}/{pict}")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(pict, fontsize=12)
            fig.suptitle(title)
    else:
        for i, ax in enumerate(axes.flatten()):
            if get_labels:
                ax.imshow(data[i][0], cmap='gray')
                ax.axis("off")
                ax.set_title(data[i][1], fontsize=12)
            else:
                ax.imshow(data[i], cmap='gray')
                ax.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()


def count_plot(x, title):
    """
    :param x: list, dataframe, np.array
    :param title: str
    :return: None
    """
    sns.countplot(x=x)
    plt.title(title)
    plt.show()


def joint_plot(df_sizes):
    """
    :param df_sizes: dataFrame
    :return: None
    """
    sns.jointplot(data=df_sizes, x="height", y="width")  # testat kind="reg" hist, reg
    plt.show()
    ax1 = sns.countplot(data=df_sizes, x="height")
    ax1.set_title("Count Height")
    ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    plt.show()
    ax2 = sns.countplot(data=df_sizes, x="width")
    ax2.set_title("Count Width")
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    plt.show()


def check_if_random_plot(df_lst, title_lst, nrows=3, ncols=1, figsize=(15, 10)):
    """
    :param df_lst: list
    :param title_lst: str
    :param nrows: int
    :param ncols: int
    :param figsize: tuple int
    :return: None
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for df, title, ax in zip(df_lst, title_lst, axes.flatten()):
        sns.scatterplot(data=df, ax=ax)
        ax.set_title(title)
    fig.suptitle("Random plot to show order in data - small data")

    plt.show()
