import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error  # Используем новую функцию
import numpy as np

def main():
    dataset = pd.read_csv('AmesHousing.csv')

    num_dataset = dataset.select_dtypes(include=['float64', 'int64'])

    # Вычисление матрицы корреляции
    num_corr = num_dataset.corr()

    # Удаление сильно коррелирующих признаков
    high_corr = set()
    for i in range(len(num_corr.columns)):
        for j in range(i):
            if abs(num_corr.iloc[i, j]) > 0.5:  # Порог 0.5
                colname = num_corr.columns[i]
                high_corr.add(colname)

    numeric_reduced = num_dataset.drop(columns=high_corr)

    # Заполнение пропущенных значений
    numeric_reduced = numeric_reduced.fillna(numeric_reduced.median())

    # Нормализация данных -  преобразования числовых данных к единому масштабу
    # в standartscaler - берется среднее и вычитается из знач и делится на стандартное отклонение
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_reduced)
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_reduced.columns)
    print(normalized_df)
    # Уменьшение размерности (PCA)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_df)

    x_pca = reduced_data[:, 0]
    y_pca = reduced_data[:, 1]
    z_target = dataset['SalePrice']

    # Построение 3D-графика
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_pca, y_pca, z_target, c=z_target, cmap='viridis')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('SalePrice')
    plt.colorbar(scatter, label='SalePrice')
    plt.show()

    # Разбиение данных
    X = normalized_df
    y = dataset['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Линейная регрессия
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f'RMSE (Linear Regression): {rmse}')

    # Lasso-регрессия
    lasso = Lasso(alpha=0.1)  # alpha - коэффициент регуляризации
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
    print(f'RMSE (Lasso): {rmse_lasso}')

    # График зависимости ошибки от коэффициента регуляризации
    alphas = np.logspace(-10, 0, 100)
    rmse_scores = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)  # Используем root_mean_squared_error
        rmse_scores.append(rmse)

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, rmse_scores, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Regularization Strength')
    plt.show()

    # Определение наиболее важных признаков
    coefficients = pd.Series(lasso.coef_, index=X_train.columns)
    important_features = coefficients[coefficients != 0].sort_values(ascending=False)
    print("Наиболее важные признаки:")
    print(important_features)

if __name__ == '__main__':
    main()