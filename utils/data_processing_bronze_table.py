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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # 转换快照日期
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # 加载数据
    csv_file_path = "data/lms_loan_daily.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())

    # 保存bronze表到datamart
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

# 在utils/data_processing_bronze_features.py中添加以下函数

def process_bronze_clickstream(snapshot_date_str, bronze_directory, spark):
    # 加载数据
    csv_file_path = "data/feature_clickstream.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # 如果数据中有snapshot_date列，则进行过滤
    if 'snapshot_date' in df.columns:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        df = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"Clickstream {snapshot_date_str} row count: {df.count()}")
    
    # 保存bronze表到datamart
    partition_name = f"bronze_clickstream_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f'Clickstream data saved to: {filepath}')
    
    return df

def process_bronze_attributes(snapshot_date_str, bronze_directory, spark):

    # 加载数据
    csv_file_path = "data/features_attributes.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # 如果数据中有snapshot_date列，则进行过滤
    if 'snapshot_date' in df.columns:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        df = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"Attributes {snapshot_date_str} row count: {df.count()}")
    
    # 保存bronze表到datamart
    partition_name = f"bronze_attributes_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f'Attributes data saved to: {filepath}')
    
    return df

def process_bronze_financials(snapshot_date_str, bronze_directory, spark):
    # 加载数据
    csv_file_path = "data/features_financials.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # 如果数据中有snapshot_date列，则进行过滤
    if 'snapshot_date' in df.columns:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        df = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"Financials {snapshot_date_str} row count: {df.count()}")
    
    # 保存bronze表到datamart
    partition_name = f"bronze_financials_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f'Financials data saved to: {filepath}')
    
    return df