import pandas as pd

parquet_path = "/Users/yahuagege/Desktop/antibody/AffinityTransformer/processed/binding/adams2017measuring/4420-fluorescein_kd-titeseq/records.parquet"
df = pd.read_parquet(parquet_path)

# 2. 检查维度（行数、列数）
print(f"=== 数据形状 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}\n")

# 3. 检查 Schema（列名、数据类型、缺失值）
print("=== 数据结构与类型 ===")
print(df.info()) 
print("\n")

# 4. 可视化查看前 5 行（在 Jupyter 中会自动渲染成漂亮的表格）
print("=== 数据样例 (前5行) ===")
# 如果列很多，可以设置展示全部列
pd.set_option('display.max_columns', None)
print(df.head())

# 5. 快速统计摘要（检查数值范围、唯一值数量等）
print("\n=== 数据统计摘要 ===")
print(df.describe(include='all'))