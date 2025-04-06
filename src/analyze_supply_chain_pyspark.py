import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.environ['HADOOP_HOME'] = r'E:\Softwares\winutils-master\winutils-master\hadoop-3.0.0'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['HADOOP_HOME'], 'bin')

# Khởi tạo SparkSession với cấu hình chi tiết
spark = SparkSession.builder \
    .appName("Enhanced Shipping Delay Analysis") \
    .config("spark.sql.debug.maxToStringFields", "1000") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

# Set logging level to ERROR to suppress warnings
spark.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu
data_path = r'D:\Tieu_Anh\bigdata\kaggle-spark-analysis\data\DataCoSupplyChainDataset.csv'
df = spark.read.csv(data_path, header=True, inferSchema=True, encoding='latin1')

# Print the initial size of the dataset
print(f"Initial dataset size: {df.count()} rows")

# Tạo cột mới để biểu thị thời gian trễ
df = df.withColumn('Shipping Delay', col('Days for shipping (real)') - col('Days for shipment (scheduled)'))

# Print the size of the dataset after adding the Shipping Delay column
print(f"Dataset size after adding Shipping Delay column: {df.count()} rows")

# Làm sạch dữ liệu
# Drop rows with missing values only in relevant columns
relevant_columns = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Order Item Quantity', 'Shipping Mode', 'Order Region', 'Shipping Delay', 'order date (DateOrders)']
df_cleaned = df.dropna(subset=relevant_columns)

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Print the size of the dataset after cleaning
print(f"Dataset size after cleaning: {df_cleaned.count()} rows")
# Save cleaned data to CSV
df.toPandas().to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)

# Check the size of the input data
if df_cleaned.count() < 10:  # Adjust the threshold as needed
    raise ValueError("Input data is too small to be split into training and test sets.")

# df_cleaned = df_cleaned.filter(col('Shipping Delay') >= 0)

# Chia dữ liệu
train_data, test_data = df_cleaned.randomSplit([0.8, 0.2], seed=42)

# Print the size of the training and test datasets
print(f"Training dataset size: {train_data.count()} rows")
print(f"Test dataset size: {test_data.count()} rows")

# Check if the training dataset is empty
if train_data.count() == 0:
    raise ValueError("Training dataset is empty. Please check the data split or the input data.")

# Phần 1: Thử nghiệm các mô hình khác nhau
# Chuẩn bị dữ liệu
indexers = []
for col_name in ['Shipping Mode', 'Order Region']:
    distinct_count = train_data.select(col_name).distinct().count()
    if distinct_count > 1:
        indexers.append(StringIndexer(inputCol=col_name, outputCol=col_name+"_index").fit(train_data))

encoders = [
    OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") 
    for col in ['Shipping Mode', 'Order Region'] if col+"_index" in train_data.columns
]

# Check if the encoded columns exist before including them in the VectorAssembler
encoded_columns = [col+"_encoded" for col in ['Shipping Mode', 'Order Region'] if col+"_encoded" in train_data.columns]

assembler = VectorAssembler(
    inputCols=encoded_columns + ['Order Item Quantity', 'Days for shipment (scheduled)'],
    outputCol="features"
)

# Định nghĩa các mô hình
models = {
    "Linear Regression": LinearRegression(featuresCol="features", labelCol="Shipping Delay", maxIter=100, regParam=0.1),
    "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="Shipping Delay", numTrees=200),
    "Gradient Boosting": GBTRegressor(featuresCol="features", labelCol="Shipping Delay", maxIter=200, maxDepth=5)
}

# Đánh giá và so sánh các mô hình
results = {}
predictions_list = []
metrics_list = []

for model_name, model in models.items():
    try:
        # Create a pipeline for each model
        pipeline = Pipeline(stages=indexers + encoders + [assembler, model])
        
        # Train the model
        trained_model = pipeline.fit(train_data)
        
        # Make predictions on the test data
        predictions = trained_model.transform(test_data)
        
        # Add model name to predictions
        predictions = predictions.withColumn("model", lit(model_name))
        predictions_list.append(predictions)
        
        # Calculate evaluation metrics
        evaluator_rmse = RegressionEvaluator(labelCol="Shipping Delay", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol="Shipping Delay", metricName="r2")
        evaluator_mae = RegressionEvaluator(labelCol="Shipping Delay", metricName="mae")
        
        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        
        # Store the metrics in a dictionary
        metrics_list.append({
            "Model": model_name,
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae
        })
        
        # Print the metrics
        print(f"""
        {model_name} Performance:
        RMSE: {rmse:.2f}
        R²: {r2:.2f}
        MAE: {mae:.2f}
        """)
    except Exception as e:
        print(f"Error with model {model_name}: {e}")

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)

# Combine all predictions into a single DataFrame
if predictions_list:
    combined_predictions = predictions_list[0]
    for pred in predictions_list[1:]:
        combined_predictions = combined_predictions.union(pred)

    # Save combined predictions to a CSV file
    combined_predictions.select("model", "features", "Shipping Delay", "prediction").toPandas().to_csv(
        os.path.join(output_dir, 'predictions.csv'), index=False
    )
else:
    print("No predictions were generated.")

# Chuyển kết quả sang DataFrame và vẽ biểu đồ
metrics_df.set_index("Model", inplace=True)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
metrics_df['RMSE'].plot(kind='bar', ax=ax[0], title='RMSE Comparison', color='skyblue')
metrics_df['R2'].plot(kind='bar', ax=ax[1], title='R² Comparison', color='lightgreen')
metrics_df['MAE'].plot(kind='bar', ax=ax[2], title='MAE Comparison', color='salmon')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png')) 




# Phần 2: Phân tích sâu hơn


# Phân tích tương quan
numeric_cols = ['Days for shipment (scheduled)', 'Days for shipping (real)', 'Order Item Quantity', 'Shipping Delay']
numeric_pd = df_cleaned.select(numeric_cols).toPandas()
plt.figure(figsize=(20, 8))
sns.heatmap(numeric_pd.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận tương quan giữa các biến số')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png')) 

# Dừng SparkSession
spark.stop()