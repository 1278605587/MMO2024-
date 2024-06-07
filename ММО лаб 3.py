import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

# 加载数据
# Загрузка данных
df = pd.read_csv('processed_dataset.csv')

# 特征缩放
# 方法1: 标准化
# Масштабирование признаков
# Метод 1: Стандартизация

scaler_standard = StandardScaler()
df_standard_scaled = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)

# 方法2: 最小-最大缩放
# Метод 2: Масштабирование Min-Max
scaler_minmax = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)

# 方法3: 离群值鲁棒缩放
# Метод 3: Масштабирование, устойчивое к выбросам
scaler_robust = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler_robust.fit_transform(df), columns=df.columns)

# 数值特征的异常值处理
# 移除异常值的方法: 使用IQR方法
# Обработка выбросов в числовых признаках
# Удаление выбросов: Использование метода IQR

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 替换异常值的方法: 使用随机森林回归
# Замена выбросов: Использование случайного леса для регрессии
def replace_outliers_with_predictions(df, column):
    model = RandomForestRegressor()
    train_data = df[df[column].notnull()]
    test_data = df[df[column].isnull()]
    
    if not test_data.empty:
        model.fit(train_data.drop(column, axis=1), train_data[column])
        predicted_values = model.predict(test_data.drop(column, axis=1))
        df.loc[df[column].isnull(), column] = predicted_values
    return df

columns_with_outliers = ['Age', 'Income']  # 这里假设这两列存在异常值
for column in columns_with_outliers:
    df = replace_outliers_with_predictions(df, column)

# 处理非标准特征
# 在这里，假设'Occupation'是非数值型非分类的特征，我们将其转换为分类特征
# Обработка нестандартных признаков
# Предположим, что 'Occupation' является нечисловым некатегориальным признаком, мы преобразуем его в категориальный

if 'Occupation' in df.columns:
    df['Occupation'] = df['Occupation'].astype('category')
    df = pd.get_dummies(df, columns=['Occupation'])

# 特征选择
# 过滤方法：使用SelectKBest和f_classif评分函数
# Выбор признаков
# Фильтрационные методы: Использование SelectKBest и f_classif

selector_kbest = SelectKBest(score_func=f_classif, k=5)
X = df.drop('Income', axis=1)
y = df['Income']
X_kbest = selector_kbest.fit_transform(X, y)

# 包装方法：使用递归特征消除 (RFE)
# Обертка: Использование рекурсивного исключения признаков (RFE)
estimator = RandomForestRegressor()
selector_rfe = RFE(estimator, n_features_to_select=5, step=1)
X_rfe = selector_rfe.fit_transform(X, y)

# 嵌入方法：使用基于惩罚项的特征选择
# Встроенные методы: Использование выбора признаков на основе регуляризации
selector_embedded = SelectFromModel(RandomForestRegressor())
X_embedded = selector_embedded.fit_transform(X, y)

# 导出处理后的数据到CSV文件
# Экспорт обработанных данных в CSV файл
processed_data_path = 'processed_data.csv'
df.to_csv(processed_data_path, index=False)

print(f"Data preprocessing is complete and the file '{processed_data_path}' has been saved.")
