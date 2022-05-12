
def initial_analyse(df):
    """
    :param df: DataFrame
    :return: print
    """
    print("info():\n", df.info(), "\n")
    print("describe():\n", df.describe(), "\n")
    print("value_counts():\n", df.value_counts(), "\n")
    # print("head():\n", df.head(), "\n")
    # print("tail():\n", df.tail(), "\n")
    print("columns:\n", df.columns, "\n")
    print("index:\n", df.index, "\n")
