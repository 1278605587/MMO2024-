import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 读取数据集
# 确保你的数据集路径和文件名正确无误
df = pd.read_csv('generated_dataset.csv')

# 缺失值处理
# 数值特征使用中位数填充
num_imputer = SimpleImputer(strategy='median')
num_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[num_columns] = num_imputer.fit_transform(df[num_columns])

# 类别特征使用众数填充
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_columns = df.select_dtypes(include=['object']).columns
df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])

# 类别特征编码
encoder = OneHotEncoder(drop='first')
encoded_columns = encoder.fit_transform(df[cat_columns]).toarray()
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(cat_columns))

# 合并编码后的类别特征和原始数据集，并移除原始类别特征列
df = df.drop(cat_columns, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# 数值特征标准化
scaler = StandardScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

# 导出处理好的数据到CSV
processed_dataset_path = 'processed_dataset.csv'
df.to_csv(processed_dataset_path, index=False)

print(f"Data preprocessing is complete and the file '{processed_dataset_path}' has been saved.")