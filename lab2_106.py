# Bai 1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset Iris
iris = load_iris(as_frame=True)
df = iris.frame

# Chọn 3 thuộc tính số
features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)"
]

# Histogram
for feature in features:
    plt.figure(figsize=(6,4))
    plt.hist(df[feature], bins=20)
    plt.title(f'Histogram - {feature}')
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# Density Plot (KDE)
plt.figure(figsize=(8,5))
for feature in features:
    sns.kdeplot(df[feature], label=feature, fill=True)

plt.legend()
plt.title("Density Plot")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df[features])
plt.title("Boxplot of Features")
plt.show()







# bai 2
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"

stock = yf.download(
    ticker,
    start="2022-01-01",
    end="2024-01-01"
)

# Moving Average
stock["MA50"] = stock["Close"].rolling(window=50).mean()
stock["MA200"] = stock["Close"].rolling(window=200).mean()

# Giá đóng cửa
plt.figure(figsize=(12,6))
plt.plot(stock.index, stock["Close"])
plt.title("Apple Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Moving Average
plt.figure(figsize=(12,6))
plt.plot(stock.index, stock["Close"], label="Close")
plt.plot(stock.index, stock["MA50"], label="MA50")
plt.plot(stock.index, stock["MA200"], label="MA200")

plt.legend()
plt.title("Apple Stock with Moving Averages")
plt.show()








# Bai 3
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot - {col}')
    plt.show()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"\n{col}")
        print("Số lượng outliers:", len(outliers))




# Bai 4
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr(numeric_only=True)

print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()