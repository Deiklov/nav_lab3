from sklearn.linear_model import LinearRegression

import pandas as pd


def read_csv(name: str) -> pd.DataFrame:
    train_data = pd.read_csv(name, sep=",", index_col=0)
    return train_data


def main():
    df = read_csv("public_data/train.csv")

    y = pd.DataFrame(df, columns=['sales'])
    X = pd.DataFrame(df, columns=df.columns[:-1])
    X = X.drop(['date'], axis=1)
    X.info()

    series = pd.Categorical(X['weather_desc'])
    print(series)
    # подготвоить все данные: колонку с датами вырезать, погоду пронумеровать от 1 до 16, город также пронумеровать
    # + стоит сделать скейлинг данных отобразить каждую колонку c цифрами на интервал 0..1

    regressor = LinearRegression()
    regressor.fit(X, y)

    df_test = read_csv("public_data/test.csv")
    X = pd.DataFrame(df_test, columns=df.columns[0:-1])
    regressor.predict(X)


if __name__ == '__main__':
    main()
