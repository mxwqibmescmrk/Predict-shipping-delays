import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, count, when, lit, avg, isnan
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.types import DoubleType, IntegerType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Cấu hình môi trường Hadoop
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

spark.sparkContext.setLogLevel("ERROR")  # Giảm log không cần thiết

# Đọc và tiền xử lý dữ liệu
DATA_PATH = r'D:\Tieu_Anh\bigdata\kaggle-spark-analysis\data\DataCoSupplyChainDataset.csv'
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True, encoding='latin1')

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# Thêm cột thời gian trễ với kiểm tra giá trị âm
df = df.withColumn('Shipping Delay', 
                  col('Days for shipping (real)') - col('Days for shipment (scheduled)'))



# Thêm phân tích chi tiết về Delivery Delay
df = df.withColumn('Delivery Delay', col('Days for shipping (real)') - col('Days for shipment (scheduled)'))

# Tính toán thống kê về Delivery Delay
delay_summary = df.select(avg("Delivery Delay")).collect()
print(f"Average Delivery Delay: {delay_summary[0][0]:.2f} days")

# Chuyển đổi sang Pandas để vẽ biểu đồ
df_pd = df.select("Delivery Delay", "Shipping Mode", "Order Region").toPandas()



# Xử lý missing values 
imputer = Imputer(
    inputCols=['Days for shipping (real)', 'Days for shipment (scheduled)'],
    outputCols=['Days for shipping (real)_imp', 'Days for shipment (scheduled)_imp']
)
df = imputer.fit(df).transform(df)

# Feature engineering: Add time-related information
df = df.withColumn('Order Year', year('order date (DateOrders)')) \
       .withColumn('Order Month', month('order date (DateOrders)'))

# Replace NaN values with 0
df = df.fillna(0)

# Remove 'Order Month' and 'Order Year' columns
df = df.drop('Order Month', 'Order Year')

# Save cleaned data to CSV
df.toPandas().to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)




# Chuẩn bị các biến đặc trưng
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
    for col in ['Shipping Mode', 'Order Region']
]

encoders = [
    OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
    for col in ['Shipping Mode', 'Order Region']
]

assembler = VectorAssembler(
    inputCols=[
        'Shipping Mode_encoded', 
        'Order Region_encoded', 
        'Order Item Quantity',
        'Days for shipment (scheduled)_imp'
        # 'Order Year',
        # 'Order Month'
    ],
    outputCol="features",
    handleInvalid="keep"  # Handle invalid values by keeping them
)

# Pipeline xử lý dữ liệu hoàn chỉnh
preprocessing_pipeline = Pipeline(stages=indexers + encoders + [assembler])
# Print the schema of the DataFrame
df.printSchema()

# Phân tích tương quan
# Check for null values in the features and label columns
df.select([count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in df.columns]).show()

relevant_columns = [
    'Days for shipping (real)', 
    'Days for shipment (scheduled)', 
    'Order Item Quantity', 
    'Shipping Mode', 
    'Order Region', 
    'Shipping Delay', 
    'order date (DateOrders)'
]
df_cleaned = df.dropna(subset=relevant_columns)

# Select numeric columns for correlation analysis
numeric_cols = ['Days for shipment (scheduled)', 'Days for shipping (real)', 'Order Item Quantity', 'Shipping Delay']

# Convert the selected numeric columns to a Pandas DataFrame
numeric_pd = df_cleaned.select(numeric_cols).toPandas()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(numeric_pd.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Variables')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png')) 

# Tinh chỉnh mô hình với Cross-Validation
def build_model(name, estimator):
    if isinstance(estimator, (RandomForestRegressor, GBTRegressor)):
        param_grid = (ParamGridBuilder()
            .addGrid(estimator.maxDepth, [3, 5])
            .addGrid(estimator.subsamplingRate, [0.8, 1.0])
            .build())
    elif isinstance(estimator, LinearRegression):
        param_grid = (ParamGridBuilder()
            .addGrid(estimator.regParam, [0.1, 0.01])
            .addGrid(estimator.elasticNetParam, [0.0, 0.5, 1.0])
            .build())
    else:
        param_grid = ParamGridBuilder().build()
    
    return CrossValidator(
        estimator=Pipeline(stages=[preprocessing_pipeline, estimator]),
        estimatorParamMaps=param_grid,
        evaluator=RegressionEvaluator(labelCol="Shipping Delay", metricName="rmse"),
        numFolds=3,
        parallelism=4
    )

models = {
    "Linear Regression": build_model("Linear Regression", LinearRegression(
        featuresCol="features", 
        labelCol="Shipping Delay",
        elasticNetParam=0.5  # Kết hợp L1/L2 regularization
    )),
    "Random Forest": build_model("Random Forest", RandomForestRegressor(
        featuresCol="features",
        labelCol="Shipping Delay",
        numTrees=200
    )),
    "Gradient Boosting": build_model("Gradient Boosting", GBTRegressor(
        featuresCol="features",
        labelCol="Shipping Delay",
        maxIter=200,
        stepSize=0.05
    ))
}

# Đánh giá và lựa chọn mô hình tốt nhất
best_models = {}
metrics_list = []
predictions_list = []
for model_name, model in models.items():
    print(f"Training {model_name} model...")
    cv_model = model.fit(df)
    best_model = cv_model.bestModel
    best_models[model_name] = best_model
    predictions = best_model.transform(df)
    
    # Tính toán các chỉ số
    evaluator = RegressionEvaluator(labelCol="Shipping Delay")
    metrics = {
        'model': model_name,
        'rmse': evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}),
        'r2': evaluator.evaluate(predictions, {evaluator.metricName: "r2"}),
        'mae': evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    }
    metrics_list.append(metrics)
    
    print(f"""
    {model_name} Performance:
    - RMSE: {metrics['rmse']:.2f}
    - R²: {metrics['r2']:.2f}
    - MAE: {metrics['mae']:.2f}
    Best Parameters:
    """)
    best_params = cv_model.bestModel.stages[-1].extractParamMap()
    for param, value in best_params.items():
        print(f"  {param.name}: {value}")
    
    # Add model name to predictions
    predictions = predictions.withColumn("model", lit(model_name))
    predictions_list.append(predictions)


# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)

# Combine all predictions into a single DataFrame
combined_predictions = predictions_list[0]
for pred in predictions_list[1:]:
    combined_predictions = combined_predictions.union(pred)

# Save combined predictions to CSV
combined_predictions.toPandas().to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Trực quan hóa nâng cao
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model.stages[-1], 'featureImportances'):
        importances = model.stages[-1].featureImportances
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [importances[i] for i in range(len(feature_names))]
        })
        plt.figure(figsize=(25, 6))
        sns.barplot(x='importance', y='feature', data=fi_df.sort_values('importance', ascending=False))
        plt.title(f'{model_name} - Feature Importance')
        plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))

# Lấy tên các đặc trưng từ VectorAssembler
feature_names = assembler.getInputCols()
for name, model in best_models.items():
    plot_feature_importance(model, feature_names, name)

# Phân tích residuals
for name, model in best_models.items():
    predictions = model.transform(df).toPandas()
    plt.figure(figsize=(20, 6))
    sns.residplot(x=predictions['prediction'], y=predictions['Shipping Delay'], lowess=True)
    plt.title(f'{name} - Residual Analysis')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(os.path.join(output_dir, f'{name}_residuals.png'))

# # Phân tích tương quan
# # Check for null or NaN values in the dataset
# # Define relevant columns for analysis
# relevant_columns = [
#     'Days for shipping (real)', 
#     'Days for shipment (scheduled)', 
#     'Order Item Quantity', 
#     'Shipping Mode', 
#     'Order Region', 
#     'Shipping Delay', 
#     'order date (DateOrders)'
# ]
# df.select([count(when(col(c).isNull() | np.isnan(col(c)), c)).alias(c) for c in df.columns]).show()

# # Ensure the label column is numeric
# df = df.withColumn("Shipping Delay", col("Shipping Delay").cast(DoubleType()))

# # Check the schema of the DataFrame
# df.printSchema()

# # Sample the dataset if it's too large
# df_sampled = df.sample(fraction=0.1, seed=42)

# # Drop rows with missing values in the relevant columns
# df_cleaned = df.dropna(subset=relevant_columns)

# # Select numeric columns for correlation analysis
# numeric_cols = ['Days for shipment (scheduled)', 'Days for shipping (real)', 'Order Item Quantity', 'Shipping Delay']

# # Compute correlation matrix using Spark
# for col1 in numeric_cols:
#     for col2 in numeric_cols:
#         if col1 != col2:
#             correlation = df_cleaned.stat.corr(col1, col2)
#             print(f"Correlation between {col1} and {col2}: {correlation}")

# # Check if the columns exist in the DataFrame and are numeric
# valid_numeric_cols = [
#     col_name for col_name in numeric_cols 
#     if col_name in [field.name for field in df_cleaned.schema.fields] 
#     and df_cleaned.schema[col_name].dataType in (DoubleType(), IntegerType())
# ]

# if not valid_numeric_cols:
#     raise ValueError("None of the specified columns are valid numeric columns in the DataFrame.")

# # Convert the selected numeric columns to a Pandas DataFrame
# numeric_pd = df_cleaned.select(valid_numeric_cols).toPandas()

# # Plot the correlation matrix as a heatmap
# plt.figure(figsize=(20, 8))
# sns.heatmap(numeric_pd.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Ma trận tương quan giữa các biến số')
# plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))  # Save the plot to the output folder
# plt.close()

# # Save the cleaned DataFrame to CSV
# df_cleaned.toPandas().to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)

# Save the entire DataFrame to CSV
df.toPandas().to_csv(os.path.join(output_dir, 'data.csv'), index=False)

# Dừng Spark
spark.stop()