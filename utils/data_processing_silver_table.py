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
from pyspark.sql.functions import col, when, greatest, least, sum, log10, lit, isnan, udf,regexp_extract, regexp_replace,trim
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.window import Window
import os
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df





def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    
    # 准备参数
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # 构造文件路径
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    
    # 加载数据
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 类型转换
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
    }
    feature_cols = [col_name for col_name in df.columns if col_name.startswith('fe_')]
    for col_name in feature_cols:
        column_type_map[col_name] = FloatType()
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
    
    # 基本数据清理：填充空值为0
    for feature in feature_cols:
        df = df.fillna({feature: 0.0})
    
    # 只保留基本列，不做聚合特征工程（避免潜在的数据泄露）
    select_columns = ["Customer_ID", "snapshot_date"] + feature_cols
    result_df = df.select(select_columns)
    
    # 保存为 Silver 表
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    result_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return result_df




#attributes
def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    
    # 准备参数
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # 连接到bronze表
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    
    # 加载数据
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 指定列及其所需数据类型的字典
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": StringType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType()
    }
    
    # 应用数据类型转换
    for column, new_type in column_type_map.items():
        if column in df.columns:  # 确保列存在
            df = df.withColumn(column, col(column).cast(new_type))
    
    # --- 1. 处理Customer_ID重复 ---
    customer_counts = df.groupBy("Customer_ID").count()
    duplicate_customers = customer_counts.filter(col("count") > 1)
    print(f"Number of duplicate Customer_IDs: {duplicate_customers.count()}")
    
    # 如果有重复，保留最新记录
    if duplicate_customers.count() > 0:
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        df_with_rank = df.withColumn("rank", F.row_number().over(window_spec))
        df = df_with_rank.filter(col("rank") == 1).drop("rank")
        print(f"After deduplication: {df.count()} records")
    
    # 首先提取数字部分
    df = df.withColumn("Age_extracted", 
                      F.regexp_extract(col("Age"), "^(\\d+)", 1))
    
    # 检查提取结果是否为空
    df = df.withColumn("Age_valid", 
                      (col("Age_extracted") != "") & 
                      (col("Age_extracted").isNotNull()))
    
    # 转换为整数并验证范围
    df = df.withColumn("Age_int", 
                      when(col("Age_valid"), 
                           col("Age_extracted").cast(IntegerType()))
                      .otherwise(None))
    
    # 验证年龄范围 (18-80)
    df = df.withColumn("Age", 
                      when((col("Age_int") >= 18) & (col("Age_int") <= 80), 
                           col("Age_int")).otherwise(None))
    df = df.withColumn("Age", 
                          F.round(col("Age")).cast(IntegerType()))
    # 删除临时列
    df = df.drop("Age_extracted", "Age_valid", "Age_int")
    
    # 检查SSN格式
    # 1. 移除连字符
    df = df.withColumn("SSN_no_dash", 
                      F.regexp_replace(col("SSN"), "-", ""))
    
    # 2. 检查是否是9位数字
    df = df.withColumn("Is_Valid_SSN",
                      when(col("SSN_no_dash").rlike("^\\d{9}$"), True)
                      .otherwise(False))
    
    # 3. 处理特殊字符情况
    df = df.withColumn("Has_Special_Chars",
                      when(col("SSN").rlike("[#F%$D@&]"), True)
                      .otherwise(False))
    
    # 4. 最终SSN处理
    df = df.withColumn("SSN_clean",
                      when(col("Is_Valid_SSN"), col("SSN_no_dash"))
                      .otherwise(None))
    
    # 检测重复的有效SSN
    valid_ssn_df = df.filter(col("Is_Valid_SSN") == True)
    if valid_ssn_df.count() > 0:
        ssn_counts = valid_ssn_df.groupBy("SSN_no_dash").count()
        duplicate_ssns = ssn_counts.filter(col("count") > 1)
        print(f"Number of duplicate valid SSNs: {duplicate_ssns.count()}")
    
    # 更新SSN字段
    df = df.withColumn("SSN", col("SSN_clean"))
    
    # 删除临时列
    df = df.drop("SSN_no_dash", "Is_Valid_SSN", "Has_Special_Chars", "SSN_clean")

    # 识别无效的职业
    df = df.withColumn("Is_Valid_Occupation",
                      when(col("Occupation").isNull(), False)
                      .when(trim(col("Occupation")) == "", False)
                      .when(col("Occupation").rlike("^[_]+$"), False)  # 只包含下划线
                      .when(trim(col("Occupation")) == "_______", False)  # 特定模式
                      .otherwise(True))
    
    # 清理职业名称 - 保留字母、数字、空格和下划线
    df = df.withColumn("Occupation_clean",
                      when(col("Is_Valid_Occupation"), trim(col("Occupation")))
                      .otherwise(None))
    
    # 更新Occupation字段
    df = df.withColumn("Occupation", col("Occupation_clean"))
    
    # 删除临时列
    df = df.drop("Is_Valid_Occupation", "Occupation_clean")
    
    # --- 5. 处理snapshot_date ---
    # 确保日期格式正确
    df = df.withColumn("snapshot_date", 
                      when(col("snapshot_date").isNull(), 
                           F.to_date(lit(snapshot_date_str), "yyyy-MM-dd"))
                      .otherwise(col("snapshot_date")))
    
    # --- 6. 保存结果 ---
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df




def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    
    # 准备参数
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # 连接到bronze表
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    
    # 加载数据
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # Customer_ID: 保持不变
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    
    # Annual_Income: 提取数字加两位小数
    df = df.withColumn("Annual_Income_raw", 
                      regexp_replace(col("Annual_Income"), "_", ""))
    df = df.withColumn("Annual_Income", 
                      F.round(regexp_extract(col("Annual_Income_raw"), r"(\d+\.?\d*)", 1).cast(DoubleType()), 2))
    df = df.drop("Annual_Income_raw")
    
    # Monthly_Inhand_Salary: 提取数字和两位小数
    df = df.withColumn("Monthly_Inhand_Salary_raw", 
                      regexp_replace(col("Monthly_Inhand_Salary").cast(StringType()), "_", ""))
    df = df.withColumn("Monthly_Inhand_Salary", 
                      F.round(regexp_extract(col("Monthly_Inhand_Salary_raw"), r"(\d+\.?\d*)", 1).cast(DoubleType()), 2))
    df = df.drop("Monthly_Inhand_Salary_raw")
    
    # Num_Bank_Accounts: 负数设为0，有效范围0-10
    df = df.withColumn("Num_Bank_Accounts", 
                      when(col("Num_Bank_Accounts") < 0, 0)
                      .when(col("Num_Bank_Accounts") > 10, 10)
                      .otherwise(col("Num_Bank_Accounts")))
    
    # Num_Credit_Card: 负数设为0，有效范围0-15
    df = df.withColumn("Num_Credit_Card", 
                      when(col("Num_Credit_Card") < 0, 0)
                      .when(col("Num_Credit_Card") > 15, 15)
                      .otherwise(col("Num_Credit_Card")))
    
    # Interest_Rate: 有效范围2%-30%
    df = df.withColumn("Interest_Rate", 
                      when((col("Interest_Rate") < 2) | (col("Interest_Rate") > 30), None)
                      .otherwise(col("Interest_Rate")))
    
    # Num_of_Loan: 提取整数，负数设为0，有效范围0-20
    df = df.withColumn("Num_of_Loan_raw", 
                      regexp_replace(col("Num_of_Loan"), "_", ""))
    df = df.withColumn("Num_of_Loan", 
                      when(regexp_extract(col("Num_of_Loan_raw"), r"(\d+)", 1).cast(IntegerType()) < 0, 0)
                      .when(regexp_extract(col("Num_of_Loan_raw"), r"(\d+)", 1).cast(IntegerType()) > 20, 20)
                      .otherwise(regexp_extract(col("Num_of_Loan_raw"), r"(\d+)", 1).cast(IntegerType())))
    df = df.drop("Num_of_Loan_raw")
    
    # Type_of_Loan: 不改变，保留原始格式用于后续特征工程
    
    # Delay_from_due_date: 保留负值，有效范围-30到180
    df = df.withColumn("Delay_from_due_date", 
                      when(col("Delay_from_due_date") > 180, 180)
                      .otherwise(col("Delay_from_due_date")))
    
    # Num_of_Delayed_Payment: 提取整数，负数设为0，有效范围0-50
    df = df.withColumn("Num_of_Delayed_Payment_raw", 
                      regexp_replace(col("Num_of_Delayed_Payment"), "_", ""))
    df = df.withColumn("Num_of_Delayed_Payment", 
                      when(regexp_extract(col("Num_of_Delayed_Payment_raw"), r"(\d+)", 1).cast(IntegerType()) < 0, 0)
                      .when(regexp_extract(col("Num_of_Delayed_Payment_raw"), r"(\d+)", 1).cast(IntegerType()) > 50, 50)
                      .otherwise(regexp_extract(col("Num_of_Delayed_Payment_raw"), r"(\d+)", 1).cast(IntegerType())))
    df = df.drop("Num_of_Delayed_Payment_raw")
    
    # Changed_Credit_Limit: 提取数字，保留正负号
    df = df.withColumn("Changed_Credit_Limit_raw", 
                      regexp_replace(col("Changed_Credit_Limit"), "_", ""))
    df = df.withColumn("Changed_Credit_Limit", 
                      F.round(regexp_extract(col("Changed_Credit_Limit_raw"), r"([+-]?\d+\.?\d*)", 1).cast(DoubleType()), 2))
    df = df.drop("Changed_Credit_Limit_raw")
    
    # Num_Credit_Inquiries: 负数设为0，有效范围0-20
    df = df.withColumn("Num_Credit_Inquiries", 
                      when(col("Num_Credit_Inquiries") < 0, 0)
                      .when(col("Num_Credit_Inquiries") > 20, 20)
                      .otherwise(col("Num_Credit_Inquiries")))
    
    # Credit_Mix: 将"_"处理为"Unknown"分类
    df = df.withColumn("Credit_Mix", 
                      when(col("Credit_Mix") == "_", "Unknown")
                      .otherwise(col("Credit_Mix")))
    
    # Outstanding_Debt: 提取数字，保留两位小数
    df = df.withColumn("Outstanding_Debt_raw", 
                      regexp_replace(col("Outstanding_Debt"), "_", ""))
    df = df.withColumn("Outstanding_Debt", 
                      F.round(regexp_extract(col("Outstanding_Debt_raw"), r"(\d+\.?\d*)", 1).cast(DoubleType()), 2))
    df = df.drop("Outstanding_Debt_raw")
    
    # Credit_Utilization_Ratio: 提取数字，有效范围0-100%
    df = df.withColumn("Credit_Utilization_Ratio", 
                      F.round(when(col("Credit_Utilization_Ratio") < 0, 0)
                            .when(col("Credit_Utilization_Ratio") > 100, 100)
                            .otherwise(col("Credit_Utilization_Ratio")), 2))
    
    # Credit_History_Age: 提取年和月
    df = df.withColumn("Credit_History_Years", 
                      regexp_extract(col("Credit_History_Age"), r"(\d+) Years?", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Months", 
                      regexp_extract(col("Credit_History_Age"), r"(\d+) Months?", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Age_Years", 
                      F.round(when(col("Credit_History_Years").isNull() & col("Credit_History_Months").isNull(), 0)
                            .when(col("Credit_History_Years").isNull(), col("Credit_History_Months") / 12)
                            .when(col("Credit_History_Months").isNull(), col("Credit_History_Years"))
                            .otherwise(col("Credit_History_Years") + (col("Credit_History_Months") / 12)), 2))
    
    # Payment_of_Min_Amount: 转为分类(Yes=1, No=0, NM=-1)
    df = df.withColumn("Payment_of_Min_Amount_Category", 
                      when(col("Payment_of_Min_Amount") == "Yes", "1")
                      .when(col("Payment_of_Min_Amount") == "No", "0")
                      .when(col("Payment_of_Min_Amount") == "NM", "-1")
                      .otherwise(None))
    
    # Total_EMI_per_month: 正值检查
    df = df.withColumn("Total_EMI_per_month", 
                      F.round(when(col("Total_EMI_per_month") < 0, 0)
                            .otherwise(col("Total_EMI_per_month")), 2))
    
    # Amount_invested_monthly: 处理"_10000_"格式，提取数字
    df = df.withColumn("Amount_invested_monthly_raw", 
                      regexp_replace(col("Amount_invested_monthly"), "_", ""))
    df = df.withColumn("Amount_invested_monthly", 
                      F.round(regexp_extract(col("Amount_invested_monthly_raw"), r"(\d+\.?\d*)", 1).cast(DoubleType()), 2))
    df = df.drop("Amount_invested_monthly_raw")
    
    # Payment_Behaviour: 保持为分类，处理乱码
    # 检查是否包含常见类别
    valid_categories = ["High_spent_Small_value_payments", "Low_spent_Large_value_payments", 
                       "Low_spent_Medium_value_payments", "High_spent_Medium_value_payments",
                       "Low_spent_Small_value_payments", "High_spent_Large_value_payments"]
    
    # 创建验证UDF函数
    is_valid_category_udf = F.udf(lambda x: x in valid_categories, "boolean")
    df = df.withColumn("Payment_Behaviour", 
                      when(is_valid_category_udf(col("Payment_Behaviour")), col("Payment_Behaviour"))
                      .otherwise(None))
    
    # Monthly_Balance: 提取数字，保留两位小数
    df = df.withColumn("Monthly_Balance_raw", 
                      regexp_replace(col("Monthly_Balance"), "_", ""))
    df = df.withColumn("Monthly_Balance", 
                      F.round(regexp_extract(col("Monthly_Balance_raw"), r"([+-]?\d+\.?\d*)", 1).cast(DoubleType()), 2))
    df = df.drop("Monthly_Balance_raw")
    
    # snapshot_date: 确保日期格式正确
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # 【已删除：收入数据有效性和债务数据有效性】
    
    # 创建"额外收入"指标，衡量除工资外的其他收入来源
    df = df.withColumn("Extra_Income_Ratio", 
                      F.round(when(col("Annual_Income").isNotNull() & col("Monthly_Inhand_Salary").isNotNull() &
                                   (col("Annual_Income") > 0) & (col("Monthly_Inhand_Salary") > 0),
                                   (col("Annual_Income") - (col("Monthly_Inhand_Salary") * 12)) / col("Annual_Income"))
                              .otherwise(None), 2))
    
    # 创建一个分类特征，表示收入差异的可能来源
    df = df.withColumn("Income_Source_Category", 
                      when(col("Extra_Income_Ratio").isNull(), "Unknown")
                      .when(col("Extra_Income_Ratio") < 0.1, "Primarily Salary")
                      .when(col("Extra_Income_Ratio") < 0.3, "Mixed Income")
                      .otherwise("Significant Non-Salary Income"))
    
    # 贷款类型数量 - 修改：同时支持","和"and"作为分隔符
    df = df.withColumn("Type_of_Loan_processed",
                      when(col("Type_of_Loan").isNull(), "")
                      .otherwise(F.regexp_replace(col("Type_of_Loan"), " and ", ",")))
    df = df.withColumn("Loan_Type_Count",
                      when(col("Type_of_Loan_processed") == "", 0)
                      .otherwise(F.size(F.split(col("Type_of_Loan_processed"), ","))))
    df = df.drop("Type_of_Loan_processed")
    
    # 财务风险基础指标：月入支出比
    df = df.withColumn("Monthly_Income_Expense_Ratio",
                     F.round(when(col("Monthly_Inhand_Salary").isNotNull() & 
                                  (col("Monthly_Inhand_Salary") > 0) &
                                  col("Total_EMI_per_month").isNotNull() &
                                  col("Amount_invested_monthly").isNotNull(),
                                  (col("Total_EMI_per_month") + col("Amount_invested_monthly")) / 
                                  col("Monthly_Inhand_Salary"))
                            .otherwise(None), 2))
    
    # 确保输出目录存在
    os.makedirs(silver_financials_directory, exist_ok=True)
    
    # 保存silver表
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df