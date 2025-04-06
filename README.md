# Enhanced Shipping Delay Analysis

This project utilizes Apache Spark to analyze a dataset obtained from Kaggle.
The analysis includes data preprocessing, model training, evaluation, and visualization of results.

## Project Structure
```
CO4033_Nhom18
├── data
│   ├── DataCoSupplyChainDataset.csv
│   ├── DescriptionDataCoSupplyChain.csv
│   └── tokenized_access_logs.csv
├── src
│   ├── analyze_supply_chain_pyspark.py
│   └── output
│        ├── cleaned_data.csv
│        ├── correlation_matrix.png
│        ├── model_comparison.png
│        ├── model_metrics.csv
│        └── predictions.csv
└── README.md
```


## Requirements
- Python 3.7+
- Apache Spark 3.0+
- PySpark
- Pandas
- Seaborn
- Matplotlib
- Statsmodels (optional, for lowess smoothing in residual plots)
- setuptools (for installing some Python packages)
- Java 8/11/17
- winutils (for Windows users)


## Setup Instructions

1. **Extract the Dataset**:
   - Ensure the dataset `DataCoSupplyChainDataset.csv` is placed in the `data` directory.
   - You can download the dataset from Kaggle here: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/data

2. **Set up a virtual environment**:
   - Create and activate a virtual environment:
     ```sh
     run infolder src # cd src
     python -m venv pyspark-venv
     .\pyspark-venv\Scripts\activate  # On Windows
     # source pyspark-venv/bin/activate  # On Unix or MacOS
     ```

3. **Install Dependencies**: 
   - Install the required libraries by running:
     ```sh
     pip install -r requirements.txt
     ```

4. **Set up Hadoop environment variables**: 
   - Download Hadoop binaries and set the `HADOOP_HOME` environment variable. Add the Hadoop `bin` directory to the `PATH`.
     ```sh
     export HADOOP_HOME=/path/to/hadoop
     export PATH=$PATH:$HADOOP_HOME/bin
     ```
   <!-- ![JAVA_HOME](image.png)
   ![HADOOP_HOME](image-1.png) -->
   <!-- add path to enviroment -->
   <!-- ![PATH](image-2.png)  -->
   

5. **Ensure Java is installed**: 
   - Make sure you have Java 8, 11, or 17 installed on your machine. You can check your Java version by running:
     ```sh
     java -version
     ```

6. **Set up winutils (for Windows users)**: 
   - Download `winutils.exe` from a trusted source and place it in a `bin` directory under your Hadoop installation directory. Set the `HADOOP_HOME` environment variable to point to your Hadoop installation directory.
     ```sh
     set HADOOP_HOME=C:\path\to\hadoop
     set PATH=%PATH%;%HADOOP_HOME%\bin
     ```

7. **Set the PYTHON_PATH environment variable**: 
   - Ensure that the `PYTHON_PATH` environment variable includes the path to your Spark installation. This can be done by running:
     ```sh
     set PYTHON_PATH=E:\Softwares\spark-3.5.5-bin-hadoop3\spark-3.5.5-bin-hadoop3\python;E:\Softwares\spark-3.5.5-bin-hadoop3\spark-3.5.5-bin-hadoop3\python\lib\py4j-0.10.9.7-src.zip  # On Windows
     export PYTHON_PATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip  # On Unix or MacOS
     ```
     <!-- set PYTHON_PATH -->
     <!-- ![PYTHON_PATH](image-3.png) -->

## Usage

1. **Run the analysis script**: 
   - Execute the main script to start the analysis:
     ```sh
     python analyze_supply_chain_pyspark.py
     ```
   - You can save the ouptput to txt file to keep tracking the result instead of print directy into terminal
     ```
     python analyze_supply_chain_pyspark.py > output.txt
     ```

2. **Check the output directory for results**: 
   - The `output` directory will contain the following files:
     - `model_metrics.csv`: Evaluation metrics for each model.
     - `predictions.csv`: Predictions from each model.
     - `data.csv`: The entire dataset after preprocessing.
     - Feature importance and residual plots for each model.
     - Correlation matrix plot

## Data

The dataset used in this project is `DataCoSupplyChainDataset.csv`, which should be placed in the `data` directory.

## Models

The following models are used in the analysis:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Results

The results of the analysis include:
- Evaluation metrics (RMSE, R², MAE) for each model.
- Feature importance plots for each model.
- Residual analysis plots for each model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Apache Spark](https://spark.apache.org/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Statsmodels](https://www.statsmodels.org/)
- [DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/data)
- [winutils](https://github.com/cdarlint/winutils)
