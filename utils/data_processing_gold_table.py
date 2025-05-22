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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_gold_attributes(snapshot_date_str, silver_attributes_directory, gold_attributes_directory, spark):
    
    from pyspark.sql.functions import col, when
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 1. 添加年龄分组特征
    df = df.withColumn(
        "Age_Group",
        when(col("Age") < 25, "Young")
        .when((col("Age") >= 25) & (col("Age") < 35), "Young_Adult")
        .when((col("Age") >= 35) & (col("Age") < 50), "Middle_Aged")
        .when((col("Age") >= 50) & (col("Age") < 65), "Senior")
        .otherwise("Elderly")
    )
    
    # 2. 职业风险评分
    df = df.withColumn(
        "Occupation_Risk_Score",
        when(col("Occupation") == "Entrepreneur", 3)  # 高风险
        .when(col("Occupation") == "Writer", 3)
        .when(col("Occupation") == "Musician", 3)
        .when(col("Occupation") == "Developer", 2)  # 中风险
        .when(col("Occupation") == "Media_Manager", 2)
        .when(col("Occupation") == "Mechanic", 2)
        .when(col("Occupation") == "Manager", 2)
        .when(col("Occupation") == "Journalist", 2)
        .when(col("Occupation") == "Lawyer", 1)  # 低风险
        .when(col("Occupation") == "Architect", 1)
        .when(col("Occupation") == "Engineer", 1)
        .when(col("Occupation") == "Accountant", 1)
        .when(col("Occupation") == "Scientist", 1)
        .when(col("Occupation") == "Teacher", 1)
        .when(col("Occupation") == "Doctor", 1)
        .otherwise(2)  # 默认为中等风险
    )
    
    # 选择最终特征列
    gold_attributes_df = df.select(
        "Customer_ID", 
        "Age", 
        "Age_Group",
        "Occupation",
        "Occupation_Risk_Score"
    )
    
    # save gold table
    partition_name = "gold_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_attributes_directory + partition_name
    gold_attributes_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return gold_attributes_df


def process_gold_financials(snapshot_date_str, silver_financials_directory, gold_financials_directory, spark):
    
    from pyspark.sql.functions import col, when, split, size, explode, regexp_replace, log
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 债务收入比(DTI)
    df = df.withColumn(
        "Debt_To_Income_Ratio",
        when(
            (col("Annual_Income").isNotNull()) & (col("Annual_Income") > 0) & 
            (col("Outstanding_Debt").isNotNull()),
            col("Outstanding_Debt") / col("Annual_Income")
        ).otherwise(None)
    )
    
    # 信用行为风险评分
    df = df.withColumn(
        "Credit_Behavior_Score",
        (
            when(col("Credit_Mix") == "Good", 3)
            .when(col("Credit_Mix") == "Standard", 2)
            .when(col("Credit_Mix") == "Bad", 1)
            .otherwise(1.5)  # 未知信用评分
        ) * 0.3 +
        (
            when(col("Payment_of_Min_Amount_Category") == "1", 3)  # Yes
            .when(col("Payment_of_Min_Amount_Category") == "0", 1)  # No
            .when(col("Payment_of_Min_Amount_Category") == "-1", 2)  # NM
            .otherwise(1.5)  # 未知付款行为
        ) * 0.3 +
        (
            when(col("Credit_Utilization_Ratio") < 30, 3)  # 低利用率
            .when(col("Credit_Utilization_Ratio") < 70, 2)  # 中等利用率
            .otherwise(1)  # 高利用率
        ) * 0.4
    )
    
    # 贷款类型处理 - One-Hot编码
    if "Type_of_Loan" in df.columns:
        # 创建贷款类型数组列，同时处理逗号和"and"分隔符
        df = df.withColumn(
            "loan_types_processed",
            regexp_replace(col("Type_of_Loan"), " and ", ",")
        )
        df = df.withColumn(
            "loan_types_array", 
            split(col("loan_types_processed"), ",")
        )
        
        # 展开并计算每种贷款类型的频率
        loan_types_exploded = df.select(
            explode(col("loan_types_array")).alias("loan_type")
        )
        
        loan_type_counts = loan_types_exploded.groupBy("loan_type").count()
        
        # 只保留出现频率较高的贷款类型（例如至少10次）
        threshold = 10  # 可以根据数据规模调整
        frequent_types = loan_type_counts.filter(col("count") >= threshold).orderBy(col("count").desc())
        
        # 收集常见贷款类型
        loan_types = [row.loan_type.strip() for row in frequent_types.collect() if row.loan_type]
        
        # 为每种常见贷款类型创建One-Hot编码列
        for loan_type in loan_types:
            if loan_type:  # 跳过空值
                clean_name = loan_type.replace(" ", "_").replace("-", "_")
                column_name = f"Loan_{clean_name}"
                df = df.withColumn(
                    column_name,
                    when(col("Type_of_Loan").isNull(), 0)
                    .when(col("Type_of_Loan").contains(loan_type), 1)
                    .otherwise(0)
                )
        
        # 删除临时列
        df = df.drop("loan_types_processed")
    
    # 选择最终特征列 - 保留绝大部分特征
    # 排除一些临时或原始文本列
    exclude_cols = ["loan_types_array", "Credit_History_Years", "Credit_History_Months"]
    select_cols = [c for c in df.columns if c not in exclude_cols]
    
    gold_financials_df = df.select(select_cols)
    
    # save gold table
    partition_name = "gold_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_financials_directory + partition_name
    gold_financials_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return gold_financials_df


def process_gold_clickstream(snapshot_date_str, silver_clickstream_directory, gold_clickstream_directory, spark):
    
    from pyspark.sql.functions import col
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 提取所有fe_特征列
    feature_cols = [c for c in df.columns if c.startswith('fe_')]
    
    # 选择最终特征列（不使用PCA）
    select_cols = ["Customer_ID", "snapshot_date"]
    select_cols.extend(feature_cols)
    
    gold_clickstream_df = df.select(select_cols)
    
    # save gold table
    partition_name = "gold_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_clickstream_directory + partition_name
    gold_clickstream_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return gold_clickstream_df


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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_gold_attributes(snapshot_date_str, silver_attributes_directory, gold_attributes_directory, spark):
    
    from pyspark.sql.functions import col, when
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 1. 添加年龄分组特征
    df = df.withColumn(
        "Age_Group",
        when(col("Age") < 25, "Young")
        .when((col("Age") >= 25) & (col("Age") < 35), "Young_Adult")
        .when((col("Age") >= 35) & (col("Age") < 50), "Middle_Aged")
        .when((col("Age") >= 50) & (col("Age") < 65), "Senior")
        .otherwise("Elderly")
    )
    
    # 2. 职业风险评分
    df = df.withColumn(
        "Occupation_Risk_Score",
        when(col("Occupation") == "Entrepreneur", 3)  # 高风险
        .when(col("Occupation") == "Writer", 3)
        .when(col("Occupation") == "Musician", 3)
        .when(col("Occupation") == "Developer", 2)  # 中风险
        .when(col("Occupation") == "Media_Manager", 2)
        .when(col("Occupation") == "Mechanic", 2)
        .when(col("Occupation") == "Manager", 2)
        .when(col("Occupation") == "Journalist", 2)
        .when(col("Occupation") == "Lawyer", 1)  # 低风险
        .when(col("Occupation") == "Architect", 1)
        .when(col("Occupation") == "Engineer", 1)
        .when(col("Occupation") == "Accountant", 1)
        .when(col("Occupation") == "Scientist", 1)
        .when(col("Occupation") == "Teacher", 1)
        .when(col("Occupation") == "Doctor", 1)
        .otherwise(2)  # 默认为中等风险
    )
    
    # 选择最终特征列
    gold_attributes_df = df.select(
        "Customer_ID", 
        "Age", 
        "Age_Group",
        "Occupation",
        "Occupation_Risk_Score"
    )
    
    # save gold table
    partition_name = "gold_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_attributes_directory + partition_name
    gold_attributes_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return gold_attributes_df


def process_gold_financials(snapshot_date_str, silver_financials_directory, gold_financials_directory, spark):
    
    from pyspark.sql.functions import col, when, split, size, explode, regexp_replace, log
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 债务收入比(DTI)
    df = df.withColumn(
        "Debt_To_Income_Ratio",
        when(
            (col("Annual_Income").isNotNull()) & (col("Annual_Income") > 0) & 
            (col("Outstanding_Debt").isNotNull()),
            col("Outstanding_Debt") / col("Annual_Income")
        ).otherwise(None)
    )
    
    # 信用行为风险评分
    df = df.withColumn(
        "Credit_Behavior_Score",
        (
            when(col("Credit_Mix") == "Good", 3)
            .when(col("Credit_Mix") == "Standard", 2)
            .when(col("Credit_Mix") == "Bad", 1)
            .otherwise(1.5)  # 未知信用评分
        ) * 0.3 +
        (
            when(col("Payment_of_Min_Amount_Category") == "1", 3)  # Yes
            .when(col("Payment_of_Min_Amount_Category") == "0", 1)  # No
            .when(col("Payment_of_Min_Amount_Category") == "-1", 2)  # NM
            .otherwise(1.5)  # 未知付款行为
        ) * 0.3 +
        (
            when(col("Credit_Utilization_Ratio") < 30, 3)  # 低利用率
            .when(col("Credit_Utilization_Ratio") < 70, 2)  # 中等利用率
            .otherwise(1)  # 高利用率
        ) * 0.4
    )
    
    # 贷款类型处理 - One-Hot编码
    if "Type_of_Loan" in df.columns:
        # 创建贷款类型数组列，同时处理逗号和"and"分隔符
        df = df.withColumn(
            "loan_types_processed",
            regexp_replace(col("Type_of_Loan"), " and ", ",")
        )
        df = df.withColumn(
            "loan_types_array", 
            split(col("loan_types_processed"), ",")
        )
        
        # 展开并计算每种贷款类型的频率
        loan_types_exploded = df.select(
            explode(col("loan_types_array")).alias("loan_type")
        )
        
        loan_type_counts = loan_types_exploded.groupBy("loan_type").count()
        
        # 只保留出现频率较高的贷款类型（例如至少10次）
        threshold = 10  # 可以根据数据规模调整
        frequent_types = loan_type_counts.filter(col("count") >= threshold).orderBy(col("count").desc())
        
        # 收集常见贷款类型
        loan_types = [row.loan_type.strip() for row in frequent_types.collect() if row.loan_type]
        
        # 为每种常见贷款类型创建One-Hot编码列
        for loan_type in loan_types:
            if loan_type:  # 跳过空值
                clean_name = loan_type.replace(" ", "_").replace("-", "_")
                column_name = f"Loan_{clean_name}"
                df = df.withColumn(
                    column_name,
                    when(col("Type_of_Loan").isNull(), 0)
                    .when(col("Type_of_Loan").contains(loan_type), 1)
                    .otherwise(0)
                )
        
        # 删除临时列
        df = df.drop("loan_types_processed")
    
    # 选择最终特征列 - 保留绝大部分特征
    # 排除一些临时或原始文本列
    exclude_cols = ["loan_types_array", "Credit_History_Years", "Credit_History_Months"]
    select_cols = [c for c in df.columns if c not in exclude_cols]
    
    gold_financials_df = df.select(select_cols)
    
    # save gold table
    partition_name = "gold_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_financials_directory + partition_name
    gold_financials_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return gold_financials_df


def process_gold_clickstream(snapshot_date_str, silver_clickstream_directory, gold_clickstream_directory, spark):
    
    from pyspark.sql.functions import col
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # 提取所有fe_特征列
    feature_cols = [c for c in df.columns if c.startswith('fe_')]
    
    # 选择最终特征列（不使用PCA）
    select_cols = ["Customer_ID", "snapshot_date"]
    select_cols.extend(feature_cols)
    
    gold_clickstream_df = df.select(select_cols)
    
    # save gold table
    partition_name = "gold_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_clickstream_directory + partition_name
    gold_clickstream_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return gold_clickstream_df


def create_gold_combined_features(snapshot_date_str, gold_attributes_directory, gold_financials_directory, gold_clickstream_directory, gold_combined_directory, spark):

    from pyspark.sql.functions import col, count, when, isnan
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # read gold attributes data
    attributes_partition_name = "gold_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    attributes_filepath = gold_attributes_directory + attributes_partition_name
    gold_attributes_df = spark.read.parquet(attributes_filepath)
    print('loaded attributes from:', attributes_filepath, 'row count:', gold_attributes_df.count())
    
    # read gold financials data
    financials_partition_name = "gold_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    financials_filepath = gold_financials_directory + financials_partition_name
    gold_financials_df = spark.read.parquet(financials_filepath)
    print('loaded financials from:', financials_filepath, 'row count:', gold_financials_df.count())
    
    # read gold clickstream data
    clickstream_partition_name = "gold_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    clickstream_filepath = gold_clickstream_directory + clickstream_partition_name
    gold_clickstream_df = spark.read.parquet(clickstream_filepath)
    print('loaded clickstream from:', clickstream_filepath, 'row count:', gold_clickstream_df.count())
    
    # 1. 首先将财务和点击流数据基于Customer_ID和snapshot_date合并 (使用left join)
    if gold_clickstream_df is not None and gold_clickstream_df.count() > 0:
        time_based_df = gold_financials_df.join(
            gold_clickstream_df,
            on=["Customer_ID", "snapshot_date"],
            how="left"  # 使用left join以保留所有金融记录
        )
    else:
        time_based_df = gold_financials_df
    
    # 2. 将时间相关数据与属性数据基于仅Customer_ID合并 (使用left join)
    combined_df = time_based_df.join(
        gold_attributes_df,
        on=["Customer_ID"],
        how="left"  # 使用left join以保留所有时间相关记录
    )
    
    # 识别非特征列、字符串列和数值列
    non_feature_cols = ["Customer_ID", "snapshot_date"]
    all_cols = [c for c in combined_df.columns if c not in non_feature_cols]
    
    # 检查每列的数据类型
    categorical_cols = []
    numeric_cols = []
    
    for col_name in all_cols:
        # 根据数据类型分类
        if combined_df.schema[col_name].dataType.simpleString() == 'string':
            categorical_cols.append(col_name)
        else:
            numeric_cols.append(col_name)
    
    print(f"Category: {categorical_cols}")
    print(f"Numeric: {numeric_cols}")
    
    # 检查是否有足够的特征
    if len(numeric_cols) == 0 and len(categorical_cols) == 0:
        raise ValueError("没有可用特征! 所有特征列都是空值或已被删除。")
    
    # 填充数值列中的null值
    for num_col in numeric_cols:
        combined_df = combined_df.fillna({num_col: 0})
    
    # 准备分类特征索引器 - 将StringIndexer结果转换为整数类型
    indexers = []
    for cat_col in categorical_cols:
        # 检查列中的非空值
        non_null_count = combined_df.filter(col(cat_col).isNotNull()).count()
        if non_null_count > 0:
            indexer = StringIndexer(
                inputCol=cat_col, 
                outputCol=f"{cat_col}_index", 
                handleInvalid="keep"
            )
            indexers.append(indexer)
    
    # 只有当有分类特征时才运行索引管道
    if indexers:
        indexer_pipeline = Pipeline(stages=indexers)
        fitted_pipeline = indexer_pipeline.fit(combined_df)
        combined_df = fitted_pipeline.transform(combined_df)
        
        # 将StringIndexer的输出转换为IntegerType
        for cat_col in categorical_cols:
            index_col = f"{cat_col}_index"
            if index_col in combined_df.columns:
                combined_df = combined_df.withColumn(index_col, col(index_col).cast(IntegerType()))
    
    print(f"Combined data row count: {combined_df.count()}")
    
    # 删除不需要的列
    columns_to_drop = [
        "Type_of_Loan_index",
        "Payment_of_Min_Amount_Category_index", 
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
        "Payment_of_Min_Amount_Category",
        "Credit_History_Age_index",
        "Age_Group",
        "Occupation"
    ]
    
    # 只删除存在的列
    existing_columns_to_drop = [col_name for col_name in columns_to_drop if col_name in combined_df.columns]
    if existing_columns_to_drop:
        combined_df = combined_df.drop(*existing_columns_to_drop)
        print(f"Dropped columns: {existing_columns_to_drop}")
    
    print(f"Final combined data row count: {combined_df.count()}")
    print(f"Final combined data column count: {len(combined_df.columns)}")
    
    # save gold combined table
    partition_name = "gold_combined_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_combined_directory + partition_name
    combined_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return combined_df, None