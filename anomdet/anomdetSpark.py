import os
import findspark
findspark.init()
import pyspark
from pyspark import SQLContext
from datetime import datetime
import sys

# sc = pyspark.SparkContext(master='spark://192.168.11.239:7077', appName='comparison')

def getData(df):
    
    df = df.select(df["Longitude"])
    df = df.na.drop()
    df = df.sort("Longitude", ascending=False)

    mLength = int(df.count()/2)
    iqLength = int(mLength/2)
    
    # Calculating mean value and IQRN
    median = float(df.collect()[mLength][0])
    IQ1 = float(df.collect()[iqLength][0])
    IQ3 = float(df.collect()[mLength + iqLength][0])
    IQRN = IQ3 - IQ1
    
    return df, median, IQRN

def getOutliers(x, median, IQRN):
    
    testS = (x - median)/IQRN
    
    if abs(testS) > 3:
        return (testS, x)

def main(sc, sqlContext):
    df = sqlContext.read.csv("/datasets/crimes.csv", header='true')
    
    df, median, IQRN = getData(df)
    
    pddf = df.toPandas()
    
    vals = pddf["Longitude"].apply(lambda x: getOutliers(float(x), median, IQRN))
    print(vals.value_counts().index)
    return vals.value_counts().index

if __name__ == "__main__":
    
    start_time = datetime.now()  
    
    sc = pyspark.SparkContext(master='spark://192.168.11.239:7077', appName='comparison')

    sqlContext = SQLContext(sc)
    
    main(sc, sqlContext)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    output1 = "tid: " + str(elapsed_time)
    sys.stderr.write(output1)
    
    

#         # create Spark context with necessary configuration
#         sc = SparkContext("local","anomdetSpark")

#         # read data from text file and split each line into words
#         words = sc.textFile("/datasets/traffic_cleaned.txt").flatMap(lambda line: line.split(" "))

#         # count the occurrence of each word
#         wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)

#         # save the counts to output
#         wordCounts.saveAsTextFile("/test/output1")
