import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Dates to process:", dates_str_lst)

# 创建Bronze层目录结构
base_directory = "datamart/bronze/"
bronze_lms_directory = os.path.join(base_directory, "lms/")
bronze_clickstream_directory = os.path.join(base_directory, "clickstream/")
bronze_attributes_directory = os.path.join(base_directory, "attributes/")
bronze_financials_directory = os.path.join(base_directory, "financials/")

# 确保所有目录存在
for directory in [bronze_lms_directory, bronze_clickstream_directory, 
                 bronze_attributes_directory, bronze_financials_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 创建Silver层目录结构
base_directory = "datamart/silver/"
silver_loan_daily_directory = os.path.join(base_directory, "loan_daily/")
silver_clickstream_directory = os.path.join(base_directory, "clickstream_daily/")
silver_attributes_directory = os.path.join(base_directory, "attributes_daily/")
silver_financials_directory = os.path.join(base_directory, "financials_daily/")

# 确保所有Silver目录存在
for directory in [silver_loan_daily_directory, silver_clickstream_directory, 
                 silver_attributes_directory, silver_financials_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 创建Gold层目录结构
base_directory = "datamart/gold/"
gold_label_store_directory = os.path.join(base_directory, "label_store/")
gold_clickstream_directory = os.path.join(base_directory, "clickstream/")
gold_attributes_directory = os.path.join(base_directory, "attributes/")
gold_financials_directory = os.path.join(base_directory, "financials/")
gold_combined_directory = os.path.join(base_directory, "combined/")

# 确保所有Gold目录存在
for directory in [gold_label_store_directory, gold_clickstream_directory, 
                 gold_attributes_directory, gold_financials_directory, gold_combined_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 导入处理函数
from utils.data_processing_bronze_table import (
    process_bronze_table,
    process_bronze_clickstream, 
    process_bronze_attributes,
    process_bronze_financials
)

print("\n===== PROCESSING BRONZE LAYER =====")

# 对每个日期执行数据处理
for date_str in dates_str_lst:
    print(f"\nProcessing bronze data for date: {date_str}")
    
    # 处理贷款数据
    process_bronze_table(date_str, bronze_lms_directory, spark)
    
    # 处理点击流、属性和财务数据
    process_bronze_clickstream(date_str, bronze_clickstream_directory, spark)
    process_bronze_attributes(date_str, bronze_attributes_directory, spark)
    process_bronze_financials(date_str, bronze_financials_directory, spark)
    
    print(f"Completed processing all bronze datasets for date: {date_str}")

print("\nAll bronze tables created successfully!")

print("\n===== PROCESSING SILVER LAYER =====")

# 对每个日期执行Silver层处理
for date_str in dates_str_lst:
    print(f"\nProcessing silver data for date: {date_str}")
    
    # 处理贷款数据
    utils.data_processing_silver_table.process_silver_table(
        date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    
    # 处理点击流数据
    utils.data_processing_silver_table.process_silver_clickstream(
        date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)
    
    # 处理属性数据
    utils.data_processing_silver_table.process_silver_attributes(
        date_str, bronze_attributes_directory, silver_attributes_directory, spark)
    
    # 处理财务数据
    utils.data_processing_silver_table.process_silver_financials(
        date_str, bronze_financials_directory, silver_financials_directory, spark)
    
    print(f"Completed processing all silver datasets for date: {date_str}")

print("\nAll silver tables created successfully!")

print("\n===== PROCESSING GOLD LAYER =====")

# 存储每个日期的金层数据框
gold_labels_dfs = {}
gold_attributes_dfs = {}
gold_financials_dfs = {}
gold_clickstream_dfs = {}
gold_combined_dfs = {}

# 对每个日期执行Gold层处理
for date_str in dates_str_lst:
    print(f"\nProcessing gold data for date: {date_str}")
    
    # 处理标签数据
    gold_labels_df = utils.data_processing_gold_table.process_labels_gold_table(
        date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6)
    gold_labels_dfs[date_str] = gold_labels_df
    
    # 处理属性数据 - 使用新的参数格式
    gold_attributes_df = utils.data_processing_gold_table.process_gold_attributes(
        date_str, silver_attributes_directory, gold_attributes_directory, spark)
    gold_attributes_dfs[date_str] = gold_attributes_df
    
    # 处理财务数据 - 使用新的参数格式
    gold_financials_df = utils.data_processing_gold_table.process_gold_financials(
        date_str, silver_financials_directory, gold_financials_directory, spark)
    gold_financials_dfs[date_str] = gold_financials_df
    
    # 处理点击流数据 - 使用新的参数格式
    gold_clickstream_df = utils.data_processing_gold_table.process_gold_clickstream(
        date_str, silver_clickstream_directory, gold_clickstream_directory, spark)
    gold_clickstream_dfs[date_str] = gold_clickstream_df
    
    # 合并特征 - 使用新的参数格式
    gold_combined_df, feature_importance = utils.data_processing_gold_table.create_gold_combined_features(
        date_str, gold_attributes_directory, gold_financials_directory, 
        gold_clickstream_directory, gold_combined_directory, spark)
    gold_combined_dfs[date_str] = gold_combined_df
    
    print(f"Completed processing all gold datasets for date: {date_str}")

print("\nAll gold tables created successfully!")


# print("\n===== 导出CSV文件 =====")

# from datetime import datetime

# try:
#     # 1. 导出标签数据为CSV
#     print("正在导出标签数据...")
#     labels_pandas_df = labels_df.toPandas()
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     labels_filename = f'gold_labels_{timestamp}.csv'
#     labels_pandas_df.to_csv(labels_filename, index=False, encoding='utf-8')
#     print(f"标签数据已保存为: {labels_filename}")
#     print(f"标签文件包含 {len(labels_pandas_df)} 行和 {len(labels_pandas_df.columns)} 列")
    
#     # 2. 导出合并的Gold数据为CSV
#     print("正在导出合并的Gold数据...")
#     combined_pandas_df = combined_df.toPandas()
#     combined_filename = f'gold_combined_all_dates_{timestamp}.csv'
#     combined_pandas_df.to_csv(combined_filename, index=False, encoding='utf-8')
#     print(f"合并Gold数据已保存为: {combined_filename}")
#     print(f"合并文件包含 {len(combined_pandas_df)} 行和 {len(combined_pandas_df.columns)} 列")
    
#     print("\n所有CSV文件导出完成！")
#     print(f"文件列表:")
#     print(f"- {labels_filename}")
#     print(f"- {combined_filename}")
    
# except Exception as e:
#     print(f"导出CSV文件时出现错误: {e}")