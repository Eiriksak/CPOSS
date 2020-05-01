import findspark
findspark.init()
import pyspark
from pyspark import SQLContext
from pyspark import SparkContext

SparkContext.setSystemProperty('spark.cleaner.periodicGC.interval', '2')
SparkContext.setSystemProperty('spark.executor.memory', '2400m')
SparkContext.setSystemProperty('spark.driver.cores', '2')
SparkContext.setSystemProperty('spark.driver.memory', '2g')
SparkContext.setSystemProperty("spark.driver.maxResultSize", "2g")

sc = pyspark.SparkContext(master='spark://192.168.11.239:7077', appName='pipeline_tests')
sqlContext = SQLContext(sc)

from pyspark.sql.types import StringType
from datetime import datetime
import pyspark.sql.functions as F #avoid conflicts with regular python functions
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler 
from pyspark.ml.feature import PCA, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
import numpy as np
import time
from pipeline_tester import PipelineTester



######CLEAN
df = sqlContext.read.csv("/datasets/crimes.csv", header='true')
#Define date derivatives
df = (df
       .withColumn('Timestamps', F.to_timestamp("Date", 'MM/dd/yyyy hh:mm:ss a'))
       .withColumn('Day', F.to_date("Date", 'MM/dd/yyyy hh:mm:ss a'))
       .withColumn("Month", F.month("Day"))
       .withColumn("Hour", F.hour("Timestamps"))
       .withColumn("Minute", F.minute("Timestamps"))
       .withColumn("DayOfMonth", F.dayofmonth("Day"))
       .withColumn("DayOfYear", F.dayofyear("Day"))
       .withColumn("DayOfWeek", F.dayofweek("Day"))
       .withColumn('WeekOfYear', F.weekofyear("Day"))
       .withColumn('Quarter', F.quarter("Timestamps"))
       
      )

cols = ["Day","Year","Month","Hour","Minute","DayOfMonth","DayOfYear","DayOfWeek","WeekOfYear","Quarter",
       "District","Primary Type"]

df = df.select(*cols)

#Rename Primary Types with less than 1% share OTHER CRIMES
def least_frequent_columns(df,threshold=0.01):
    res = df.groupBy("Primary Type").count()\
                            .withColumn('tot',F.lit(df.count()))\
                            .withColumn('frac',F.expr('count/tot'))\
                            .filter('frac<'+str(threshold))\
                            .select("Primary Type")\
                            .rdd.flatMap(lambda x: x)\
                            .collect()
    return res


@udf(StringType())
def renamer(label):
    if label in lfc:
        return "OTHER CRIMES"
    else:
        return label
    
    
lfc = least_frequent_columns(df)

#Downsample the data until it contains approx 150k samples
fractions = df.select("Primary Type").distinct().withColumn("fraction", F.lit(0.02)).rdd.collectAsMap()

#Rename, cast, downsample, filter and drop columns
df = (df.withColumn('y', renamer(F.col('Primary Type')))\
      .drop("Primary Type")\
      .withColumn("_Year", df.Year.cast('integer'))\
      .drop(*["Day","Year"])\
      .withColumnRenamed("_Year","Year")\
      .filter(F.col("Year") < 2020)
      .sampleBy("Primary Type", fractions, 42)
      .repartition(216, 'Year','Month')) #Partition each year/month combo

######EDN CLEAN


######PIPELINES
#=========== BASE OF THE PIPELINE ===========#

categorical_cols = ["District"]

indexers = [ StringIndexer(inputCol=cat_col, outputCol="{}_idx".format(cat_col),
                           handleInvalid = 'skip') for cat_col in categorical_cols] 

target_indexer = [ StringIndexer(inputCol = 'y', outputCol = 'target', handleInvalid = 'skip')]


encoders = [OneHotEncoder(dropLast=True,inputCol=idx.getOutputCol(), 
    outputCol="{}_catVec".format(idx.getOutputCol())) for idx in indexers]

#=========== END BASE ===========#




#=========== FEAUTURE COLUMNS ===========#

#All
fc_1 = ["Year","Month","Hour","Minute","DayOfMonth",
                "DayOfYear","DayOfWeek","WeekOfYear","Quarter"] \
+ [enc.getOutputCol() for enc in encoders]


#No yearly perspective
fc_2 = ["Hour","Month","DayOfWeek"] + [enc.getOutputCol() for enc in encoders]


#With yearly perspective
fc_3 = ["Year","Hour","Month","DayOfWeek"] + [enc.getOutputCol() for enc in encoders]


#No seasonal perspective 
fc_4 = ["Year","Hour","DayOfWeek"] + [enc.getOutputCol() for enc in encoders]


#Precise daily perspective
fc_5 = ["Hour","Minute"] + [enc.getOutputCol() for enc in encoders]


#Unprecise daily perspective
fc_6 = ["Hour"] + [enc.getOutputCol() for enc in encoders]

fcs = [fc_1,fc_2,fc_3,fc_4,fc_5,fc_6]
#=========== END FC ===========#


standard_scaler = StandardScaler(inputCol="Features", outputCol="scaledFeatures", withStd=False, withMean=True)
min_max_scaler = MinMaxScaler(inputCol="Features", outputCol="scaledFeatures")
max_abs_scaler = MaxAbsScaler(inputCol="Features", outputCol="scaledFeatures")

norm_standard_scaler = StandardScaler(inputCol="normFeatures", outputCol="scaledFeatures", withStd=False, withMean=True)
norm_min_max_scaler = MinMaxScaler(inputCol="normFeatures", outputCol="scaledFeatures")
norm_max_abs_scaler = MaxAbsScaler(inputCol="normFeatures", outputCol="scaledFeatures")

normalizer = Normalizer(inputCol="Features", outputCol="normFeatures")

######END PIPELINE


from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier, DecisionTreeClassifier
import json

#Create a pipeline testing object and run tests
pt = PipelineTester(df)

#Logistic regression
pt.lr(base= indexers + encoders + target_indexer, fcs=fcs)
pt.lr(base= indexers + encoders + target_indexer, scaler = [standard_scaler], fcs=fcs[1:4])
pt.lr(base= indexers + encoders + target_indexer, scaler = [min_max_scaler], fcs=fcs[1:4])
pt.lr(base= indexers + encoders + target_indexer, normalizer = [normalizer], scaler=[norm_min_max_scaler], fcs=fcs[1:4])
pt.lr(base= indexers + encoders + target_indexer, normalizer = [normalizer], scaler=[norm_standard_scaler], fcs=fcs[1:4])
pt.lr(base= indexers + encoders + target_indexer, normalizer = [normalizer], scaler=[norm_max_abs_scaler], fcs=fcs[1:4])
pt.lr(base= indexers + encoders + target_indexer, scaler = [standard_scaler], pca=[3,10],  fcs=fcs[:2]) #Know this wont improve


with open('logs/lr_complete.json', 'w+') as f:
    json.dump(pt.logger["lr"], f, indent=4)

#Decision Tree
pt.dt(base= indexers + encoders + target_indexer, fcs=fcs)
pt.dt(base= indexers + encoders + target_indexer, scaler = [standard_scaler], fcs=fcs[:2])#know this has no effect
pt.dt(base= indexers + encoders + target_indexer, scaler = [min_max_scaler], fcs=fcs[:2])
pt.dt(base= indexers + encoders + target_indexer, normalizer = [normalizer], fcs=fcs[1:4])
pt.dt(base= indexers + encoders + target_indexer, scaler = [standard_scaler], pca=[3,6,10,15,20], fcs=fcs[1:4])

with open('logs/dt_complete.json', 'w+') as f:
    json.dump(pt.logger["dt"], f, indent=4)


#MLP
pt.mlp(base= indexers + encoders + target_indexer, fcs=fcs)
pt.mlp(base= indexers + encoders + target_indexer, scaler = [standard_scaler], fcs=fcs[1:4])
pt.mlp(base= indexers + encoders + target_indexer, scaler = [min_max_scaler], fcs=fcs[1:4])
pt.mlp(base= indexers + encoders + target_indexer, normalizer = [normalizer], fcs=fcs[1:4])
pt.mlp(base= indexers + encoders + target_indexer, normalizer = [normalizer], scaler=[norm_min_max_scaler], fcs=fcs[1:4])
pt.mlp(base= indexers + encoders + target_indexer, normalizer = [normalizer], scaler=[norm_standard_scaler], fcs=fcs[1:4])
pt.mlp(base= indexers + encoders + target_indexer, normalizer = [normalizer], scaler=[norm_max_abs_scaler], fcs=fcs[1:4])
pt.mlp(base= indexers + encoders + target_indexer, scaler = [standard_scaler], pca=[3,6,10,15,20],  fcs=fcs[1:4])

with open('logs/mlp_complete.json', 'w+') as f:
    json.dump(pt.logger["mlp"], f, indent=4)

#Store all
with open('logs/pipeline_tests.json', 'w+') as f:
    json.dump(pt.logger, f, indent=4)

sc.stop()
