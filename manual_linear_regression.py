import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == "__main__":

    filename = sys.argv[1]

    df = pd.read_csv(filename, header=0)


    #Q2
    df.plot(x ='year', y='days', kind='line')
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg")

    #Q3
    n= len(df)
    X= np.ones((n,2))
    X = X.astype(int)
    X[:, 1] = df['year']
    print("Q3a:")
    print(X)

    Y = df['days'].to_numpy()
    print("Q3b:")
    print(Y)

    Z = X.transpose()@ X
    print("Q3c:")
    print(Z)

    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    PI = I @ X.transpose()
    print("Q3e:")
    print(PI)

    hat_beta= PI @ Y
    print("Q3f:")
    print(hat_beta)

    x_2022= np.array([[1,2022]])
    y_test= x_2022@hat_beta
    print("Q4: " + str(y_test[0]))


    symbol = None
    if hat_beta[1] < 0:
        symbol = '<'
    elif hat_beta[1] == 0:
        symbol = '='
    else:
        symbol = '>'

    print("Q5a: " + str(symbol))

    Short_answer= 'Statistically, the number of frozen days of Lake Mendota through the years is trending down \
    with the slope of hat_beta_1 = -0.197.'
    print("Q5b: " + Short_answer)

    x_star = -hat_beta[0] / hat_beta[1]
    print("Q6a: " + str(x_star))

    answer= 'First, we are unsure if there is a linear relationship between the years and the number of frozen days. \
    Second, the trend is small enough to be questionable. Based on the trend, there will be a year from which Lake Mendota no longer freezes. \
    But the variation in the data is very large, as can be seen from the plot.\
    Therefore, there is a wide range of variation from the expected year. Also, even after the \
    expected year, the lake may still freeze, due to the large variation.'

    print("Q6b: " + answer)