# Bài 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris


# TẢI DỮ LIỆU IRIS

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# thêm cột phân loại
df["species"] = iris.target

# đổi số thành tên
df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})


# HIỂN THỊ 10 DÒNG ĐẦU

print("10 dòng dữ liệu đầu tiên:")
print(df.head(10))


# KIỂM TRA KÍCH THƯỚC DỮ LIỆU

print("\nSố lượng bản ghi và thuộc tính:")
print(df.shape)


# KIỂM TRA KIỂU DỮ LIỆU

print("\nKiểu dữ liệu của từng cột:")
print(df.dtypes)


# THỐNG KÊ CƠ BẢN

print("\nThống kê cơ bản:")
print(df.describe())


# GIÁ TRỊ TRUNG BÌNH

print("\nGiá trị trung bình:")
print(df.mean(numeric_only=True))

# GIÁ TRỊ LỚN NHẤT

print("\nGiá trị lớn nhất:")
print(df.max(numeric_only=True))

# GIÁ TRỊ NHỎ NHẤT

print("\nGiá trị nhỏ nhất:")
print(df.min(numeric_only=True))


# ĐỘ LỆCH CHUẨN

print("\nĐộ lệch chuẩn:")
print(df.std(numeric_only=True))


# HISTOGRAM CHO THUỘC TÍNH SỐ
df.hist(figsize=(10, 8))
plt.suptitle("Histogram các thuộc tính số")
plt.show()

# BAR CHART CHO THUỘC TÍNH PHÂN LOẠI

plt.figure(figsize=(6, 4))

df["species"].value_counts().plot(kind="bar")

plt.title("Số lượng từng loài hoa")
plt.xlabel("Loài")
plt.ylabel("Số lượng")

plt.show()







# bài 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# TẢI DỮ LIỆU
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# TẠO DỮ LIỆU THIẾU GIẢ LẬP
df.loc[5, "sepal length (cm)"] = None
df.loc[10, "petal width (cm)"] = None

# KIỂM TRA DỮ LIỆU THIẾU
print("Số lượng giá trị thiếu:")
print(df.isnull().sum())

# ĐIỀN GIÁ TRỊ TRUNG BÌNH
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nDữ liệu sau khi xử lý thiếu:")
print(df.isnull().sum())

# KIỂM TRA TRÙNG LẶP
print("\nSố dòng trùng lặp:")
print(df.duplicated().sum())

# XÓA DỮ LIỆU TRÙNG LẶP
df.drop_duplicates(inplace=True)

print("\nKích thước sau khi xóa trùng:")
print(df.shape)

# BOXPLOT TRƯỚC KHI CHUẨN HÓA
plt.figure(figsize=(10, 5))

sns.boxplot(data=df)

plt.title("Boxplot trước chuẩn hóa")

plt.show()

# CHUẨN HÓA DỮ LIỆU
scaler = StandardScaler()

numeric_cols = df.columns

df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=numeric_cols
)

# BOXPLOT SAU CHUẨN HÓA
plt.figure(figsize=(10, 5))

sns.boxplot(data=df_scaled)

plt.title("Boxplot sau chuẩn hóa")

plt.show()

# HIỂN THỊ DỮ LIỆU SAU CHUẨN HÓA
print("\nDữ liệu sau chuẩn hóa:")
print(df_scaled.head())