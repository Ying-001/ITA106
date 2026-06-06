# ==========================
# GIAI ĐOẠN 1 - EDA LEARNX
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv(r'C:\Python III\Lab1.py\learnx_user_behavior_dataset_10M\learnx_user_behavior_dataset_10M.csv')


# ==========================
# 1. THÔNG TIN DỮ LIỆU
# ==========================

print("Kích thước dữ liệu:")
print(df.shape)

print("\n5 dòng đầu:")
print(df.head())

print("\nThông tin dữ liệu:")
print(df.info())

print("\nThống kê mô tả:")
print(df.describe())

# ==========================
# 2. LÀM SẠCH DỮ LIỆU
# ==========================

print("\nGiá trị thiếu:")
print(df.isnull().sum())

print("\nSố dòng trùng lặp:")
print(df.duplicated().sum())

# Xóa dòng trùng lặp
df = df.drop_duplicates()

# ==========================
# 3. PHÂN TÍCH DỮ LIỆU
# ==========================

numeric_cols = df.select_dtypes(include=np.number).columns

# Histogram cho các thuộc tính số
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    plt.hist(df[col], bins=20)
    plt.title(f'Histogram - {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ==========================
# Density Plot (KDE)
# ==========================

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.kdeplot(df[col], fill=True)
    plt.title(f'Density Plot - {col}')
    plt.show()

# ==========================
# Boxplot
# ==========================

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.title("Boxplot các thuộc tính số")
plt.show()

# ==========================
# 4. PHÂN TÍCH HÀNH VI
# ==========================

# Thời gian học
if "study_time_hours" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df["study_time_hours"],
                 bins=30,
                 kde=True)
    plt.title("Phân phối thời gian học")
    plt.show()

# Số lần truy cập
if "weekly_visits" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df["weekly_visits"],
                 bins=20,
                 kde=True)
    plt.title("Số lần truy cập mỗi tuần")
    plt.show()

# Tỷ lệ hoàn thành
if "completion_rate" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df["completion_rate"],
                 bins=20,
                 kde=True)
    plt.title("Tỷ lệ hoàn thành khóa học")
    plt.show()

# ==========================
# 5. MA TRẬN TƯƠNG QUAN
# ==========================

plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ==========================
# 6. PHÁT HIỆN BẤT THƯỜNG
# ==========================

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"\n{col}")
    print("Số outlier:", len(outliers))

# ==========================
# 7. INSIGHT TỔNG QUAN
# ==========================

print("\n=== KẾT LUẬN ===")
print("- Xem phân phối dữ liệu qua Histogram và KDE.")
print("- Xem ngoại lệ bằng Boxplot.")
print("- Xem mối quan hệ giữa các biến bằng Heatmap.")
print("- Phát hiện người dùng có hành vi bất thường.")