"""
Run this script in order to generate probability feature vectors for the
Chicago Crimes dataset.
WARNING: This script requires a lot of memory, so we recommend to run on
only a smaller samplet subset of the original dataset. The same probabilities
will be captured anyways.
"""



import sys
import os

def configure_spark(spark_home=None, pyspark_python=None):
    spark_home = spark_home or "/path/to/default/spark/home"
    os.environ['SPARK_HOME'] = spark_home

    # Add the PySpark directories to the Python path:
    sys.path.insert(1, os.path.join(spark_home, 'python'))
    sys.path.insert(1, os.path.join(spark_home, 'python', 'pyspark'))
    sys.path.insert(1, os.path.join(spark_home, 'python', 'build'))

    # If PySpark isn't specified, use currently running Python binary:
    pyspark_python = pyspark_python or sys.executable
    os.environ['PYSPARK_PYTHON'] = pyspark_python
    
configure_spark('/usr/local/spark', '/home/ubuntu/anaconda3/envs/dat500/bin/python')


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

sc = pyspark.SparkContext(master='spark://192.168.11.239:7077', appName='type_predicter')
sqlContext = SQLContext(sc)

from pyspark.sql.types import *
from datetime import datetime
import pyspark.sql.functions as F #avoid conflicts with regular python functions
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler 
import numpy as np
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, MatrixUDT, VectorUDT, DenseMatrix, DenseVector
import time
import math
import pandas as pd


#df = sqlContext.read.csv("/datasets/district11.csv", header='true')
df = sqlContext.read.csv("/datasets/crimes.csv", header='true')
start = time.time()
#Define date derivatives
df = (df
       .withColumn('Timestamps', F.to_timestamp("Date", 'MM/dd/yyyy hh:mm:ss a'))
       .withColumn('Day', F.to_date("Date", 'MM/dd/yyyy hh:mm:ss a'))
       .withColumn("Month", F.month("Day"))
       .withColumn("Hour", F.hour("Timestamps"))
       .withColumn("DayOfYear", F.dayofyear("Day"))
       .withColumn("DayOfWeek", F.dayofweek("Day"))

       
      )

cols = ["ID","Day","Year","Month","Hour","DayOfYear","DayOfWeek","District","Primary Type"]

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

fractions = df.select("Primary Type").distinct().withColumn("fraction", F.lit(0.05)).rdd.collectAsMap()

#Rename, cast, downsample, filter and drop columns
df = (df.withColumn('y', renamer(F.col('Primary Type')))\
      .drop("Primary Type")\
      .filter( (F.col("District")!='021') & (F.col("District")!='031'))\
      .withColumn("_Year", df.Year.cast('integer'))\
      .drop(*["Year"])\
      .withColumnRenamed("_Year","Year")\
      .filter(F.col("Year") < 2020)
      .sampleBy("Primary Type", fractions, 42)
      .repartition(216, 'Year','Month')) #Partition each year/month combo

print("Finished the data cleaning in  ", round((time.time() - start),1), " seconds")

join_df = df\
.groupBy("District", "Day", "Hour","y")\
.count()\
.select(F.col("District").alias("jDistrict"), #Rename to avoid conflict on join
       F.col("Day").alias("jDay"),
       F.col("Hour").alias("jHour"),
       F.col("y").alias("jy"),
       F.col("count").alias("CountHour"))


df = df.join(join_df,\
              ([join_df.jy == df.y,\
                join_df.jDistrict == df.District,
                  join_df.jDay == df.Day,\
                join_df.jHour == df.Hour]),\
              how='left')\
.drop("jDay","jDistrict","jy","jHour")


tmp = df.dropna(how='any').select("DayOfYear","DayOfWeek","Year","Day","District",\
                     "Hour","y","CountHour")\
                .dropDuplicates()


join_df = tmp\
    .groupBy("y","District","Year")\
    .agg(F.collect_list("CountHour").alias("hc"),
        F.collect_list("DayOfWeek").alias("dow"),
        F.collect_list("DayOfYear").alias("doy"),
        F.collect_list("Hour").alias("hr"))\
    .select(
        F.col("y").alias("jy"),
        F.col("District").alias("jDistrict"),
        F.col("Year").alias("jYear"),
        F.col("hc").alias("hc"),
        F.col("dow").alias("dow"),
        F.col("doy").alias("doy"),
        F.col("hr").alias("hr"))

tmp = tmp.join(join_df,\
              ([join_df.jy == df.y,\
                join_df.jDistrict == df.District,
                join_df.jYear == df.Year]),\
              how='left')\
.drop("jYear","jDistrict","jy")


@udf(MatrixUDT())
def yearly_dayofweek_hour_matrix(doy_ar, dow_ar, hr_ar, hc_ar, dow, hr, year):
    """
    Input is all crimes of a certain type for a year, within a district
    E.g. All Narcotics crimes of 2018 in District 09
    
    params:
    doy_ar: Day of the year array
    dow_ar: Day of the week array
    hr_ar: Hour array
    hc_ar: Hour count array
    arrays should be of the same length. By stacking them on top of each other in a matrix, we get
    the day of the year, day the week, hour and amount of a certain crime by filtering on an index (column)
    
    doy: Day of the year for the incoming row
    dow: Day of the week value for the incoming row
    year: Year of the incoming row
    We should use this to filter the matrix until it only contains crimes occuring on the same day
    of the week, with +- 1 hour 
    
    returns:
    A dense, but filtered matrix containing only relevant crimes for this input
    shape: (4, num crimes)
    Example: [4,20,55], (Day of the year)
             [2,4,2],  (Day of the week)
             [22,21,23] (Hour)
             [2,1,1]   (Count)
        
    Filter on crimes at mondays 23:00 -> column 0 and 2
    Filter on crimes at mondays 00:00 -> column 2
    Filter on crimes at wednesdays 20:00 -> column 1 
    """
    dense_mat = np.matrix([doy_ar ,dow_ar ,hr_ar, hc_ar]) #Dense matrix with all crimes that year
    dow_filter = dow_filter = dense_mat[:, np.array(dense_mat[1,:]==dow).flatten()] #Only the same day of the week
    #+-1 hour
    hr_filtered = dow_filter[:, np.isin( np.array( dow_filter[2,:]).flatten(), np.array([(hr-1)%24, hr, (hr+1)%24]))]
    #Last column stores what we filtered on
    vals = np.append(np.array(hr_filtered), np.full(hr_filtered.shape[1],year).reshape(1,-1), axis=0)
    return DenseMatrix(numRows = 5 , numCols = vals.shape[1], values=vals.flatten(), isTransposed=True)



tmp = tmp.withColumn("yearly_hcount_mat",
        yearly_dayofweek_hour_matrix(
            F.col("doy"),
            F.col("dow"),
            F.col("hr"),
            F.col("hc"),
            F.col("DayOfWeek"),
            F.col("Hour"),
            F.col("Year")
        ))\
.drop("doy","dow","hr","hc")


@udf(MatrixUDT())
def all_time_mat(matrices):
    """
    Input a list of matrices for a crime type inside a district, a day of the week during an hour
    Stack them together in order to get statistics from all available year later on 
    """
    try:
        if len(list(matrices)) < 1:
            return None

        mat_array = [m.toArray() for m in list(matrices)]
        years = [mat_array[0][-1,0]]
        for yearly_matrix in mat_array[1:]:
            year = yearly_matrix[-1,0]
            if year in years:
                return None #Should be one matrix per year if done correctly
            
        max_len = max([arr.shape[1] for arr in mat_array])
        stacked_padded_matrix = np.array([np.lib.pad(arr, ((0,0), (0, max_len - arr.shape[1])),
                                                     'constant', constant_values=5000) for arr in mat_array])
        stacked_padded_matrix = stacked_padded_matrix.reshape(-1,max_len)
        return DenseMatrix(numRows = stacked_padded_matrix.shape[0],
                           numCols = stacked_padded_matrix.shape[1],
                           values = stacked_padded_matrix.flatten(),
                           isTransposed=True)
    except:
        return None



join_df = tmp.groupBy("District","y","DayOfWeek","Hour")\
    .agg(F.collect_set("yearly_hcount_mat").alias("matrices"))\
    .withColumn("all_time_mat",
                all_time_mat(
                    F.col("matrices")
                ))\
    .select(
        F.col("y").alias("jy"),
        F.col("District").alias("jDistrict"),
        F.col("DayOfWeek").alias("jDayOfWeek"),
        F.col("Hour").alias("jHour"),
        F.col("all_time_mat").alias("all_time_mat"))


tmp = tmp.join(join_df,\
              ([join_df.jy == df.y,\
                join_df.jDistrict == df.District,
                join_df.jDayOfWeek == df.DayOfWeek,
                join_df.jHour == df.Hour]),\
              how='left')\
.drop("jDayOfWeek","jDistrict","jy","jHour")


@udf(FloatType())
def all_time_avg(matrix, year, dayofyear, new_year_pred=None):
    """
    Avg all time, that time of the week, that hour
    
    params:
    matrix: All time matrix of a certain crime happening in District x on the y day of the week and z +-1 hour
    year: The year of incoming crime
    dayofyear: The day of the year of the incoming crime
    
    returns:
    Based on the year and dayofyear, it will count all occurences up until that day and divide it by
    amount of weeks since the beginning
    
    """
    try:
        crime_counter = math.ceil(dayofyear/7)
        if new_year_pred:
            crime_counter = 0
            
        if crime_counter > 52:
            crime_counter = 52
        
        if year > 2001:
            crime_counter += ((int(year)-1)-2001)*52
            
        last_slicer = 0
        total_crimes = 0
        matrix = matrix.toArray()
        for slicer in np.arange(5 , matrix.shape[0]+1, 5): #Slice year for year
            cur = matrix[last_slicer:slicer]
            cur = cur[:, np.array(cur[1,:]!=5000).flatten()] #Remove padded columns
            last_slicer = slicer
            
            #If years before, include all that year
            if cur[4,0] < int(year):
                total_crimes += np.sum(cur[3])
                
            #If the same year, include only days before
            elif cur[4,0] == int(year):
                cur = cur[:, np.array(cur[0,:]< int(dayofyear)).flatten()] 
                total_crimes += np.sum(cur[3])
                
        return float(total_crimes/crime_counter)
    except:
        return float(0)


@udf(FloatType())
def avg_past_year(matrix, year, dayofyear, new_year_pred=None):
    """
    Avg past 365 days, that day of the week, (+- 1 hour)
    
    params:
    matrix: All time matrix of a certain crime happening in District x on the y day of the week and +-1 hour
    year: The year of incoming crime
    dayofyear: The day of the year of the incoming crime
    new_year_pred: If true, this is the dayofyear prediction of a future dataframe
    
    returns:
    Based on the year and dayofyear, it will count all occurences the last 365 days and divide it by
    the amount of weeks during last 365 days (52)
    
    """
    try:

        if year == 2001:
            crime_counter = math.ceil(dayofyear/7)
            if crime_counter > 52:
                crime_counter = 52
        elif new_year_pred:
            crime_counter = math.ceil((365-dayofyear)/7)
        else:
            crime_counter = 52
            
        last_slicer = 0   
        total_crimes = 0
        matrix = matrix.toArray()
        for slicer in np.arange(5 , matrix.shape[0]+1, 5): #Slice year for year
            cur = matrix[last_slicer:slicer]
            cur = cur[:, np.array(cur[1,:]!=5000).flatten()] #Remove padded columns
            last_slicer = slicer
            
            if cur[4,0] not in [year,year-1]:
                continue
            
            #If the year before, include days after this dayofyear
            if cur[4,0] == (int(year)-1):
                cur = cur[:, np.array(cur[0,:] >= int(dayofyear)).flatten()] 
                total_crimes += np.sum(cur[3])
                
            #If the same year, include only days before
            if cur[4,0] == int(year):
                cur = cur[:, np.array(cur[0,:] < int(dayofyear)).flatten()] 
                total_crimes += np.sum(cur[3])
                
        return float(total_crimes/crime_counter)
    except:
        return float(0)


@udf(FloatType())
def avg_past_month(matrix, year, dayofyear, new_year_pred=None):
    """
    Avg past 30 days, that day of the week, (+- 1 hour)
    
    params:
    matrix: All time matrix of a certain crime happening in District x on the y day of the week and z +-1 hour
    year: The year of incoming crime
    dayofyear: The day of the year of the incoming crime
    new_year_pred: If not None, this is the dayofyear prediction of a future dataframe. Set it to the number
                    of extra days to include while computing
    
    returns:
    Based on the year and dayofyear, it will count all occurences during the last 30 days and divide it by
    amount of weeks during the last 30 days (4). If the day is within the first 30 days of a new year, it will
    look at last i remaining days from the year before as well.
    
    """
    try:
        crime_counter = 4
        remaining_days = None
        if dayofyear < 31 and not new_year_pred:
            remaining_days = 365 - np.abs(29-dayofyear) #Days to include from last year
            
        if new_year_pred:
            remaining_days = 365 - np.abs(new_year_pred-dayofyear)
            crime_counter = math.floor(np.abs(new_year_pred-dayofyear)/7)
            
        last_slicer = 0
        total_crimes = 0
        matrix = matrix.toArray()
        for slicer in np.arange(5 , matrix.shape[0]+1, 5): #Slice year for year
            cur = matrix[last_slicer:slicer]
            cur = cur[:, np.array(cur[1,:]!=5000).flatten()] #Remove padded columns
            last_slicer = slicer
            
            if remaining_days and cur[4,0] == (int(year)-1):
                cur = cur[:, np.array(cur[0,:] >= remaining_days).flatten()] 
                total_crimes += np.sum(cur[3])
                
            #If the same year, include only past 30 days
            elif cur[4,0] == int(year):
                cur = cur[:, np.isin(np.array(cur[0,:]), np.arange((dayofyear-30),dayofyear) )] 
                total_crimes += np.sum(cur[3])
                
        return float(total_crimes/crime_counter)
    except:
        return float(0)



@udf(FloatType())
def avg_last_year_months(matrix, year, dayofyear):
    """
    Avg +- 30 days, last year, that day of the week, (+- 1 hour)
    
    params:
    matrix: All time matrix of a certain crime happening in District x on the y day of the week and z +-1 hour
    year: The year of incoming crime
    dayofyear: The day of the year of the incoming crime
    
    returns:
    Based on the year and dayofyear, it will count all occurences that happened around the same time, last year
    (+- 30 days from dayofyear, year-1). Then it will divide the count by number of weeks during those
    60 days (8). If the day is within the first 30 days of a new year, it will look at last i remaining days 
    from the year before as well. If the day is within the last 30 days of the year, it will look at the j 
    remaining days from the same year as well.
    """
    try:
        crime_counter = 8
        ey_remaining_days = None
        ly_remaining_days = None
        if dayofyear < 31: #include from two years ago as well
            ey_remaining_days = 365 - np.abs(29-dayofyear) 
            
        if dayofyear > (365-30): #include from this year as well
            _rem = (365-dayofyear)
            if _rem < 0:
                _rem = 0
            ly_remaining_days = 30 - _rem
            
            
        last_slicer = 0
        total_crimes = 0
        matrix = matrix.toArray()
        for slicer in np.arange(5 , matrix.shape[0]+1, 5): #Slice year for year
            cur = matrix[last_slicer:slicer]
            cur = cur[:, np.array(cur[1,:]!=5000).flatten()] #Remove padded columns
            last_slicer = slicer
            
            if ey_remaining_days and cur[4,0] == (int(year)-2):
                cur = cur[:, np.array(cur[0,:] >= remaining_days).flatten()] 
                total_crimes += np.sum(cur[3])
                

            elif ly_remaining_days and cur[4,0] == year:
                cur = cur[:, np.array(cur[0,:] <= int(ly_remaining_days)).flatten()] 
                total_crimes += np.sum(cur[3])
                
                
            elif cur[4,0] == (int(year)-1):
                cur = cur[:, np.isin(np.array(cur[0,:]), np.arange((dayofyear-30),(dayofyear+30)))] 
                total_crimes += np.sum(cur[3])
                
        return float(total_crimes/crime_counter)
    except:
        return float(0)


tmp = (tmp\
    .withColumn("avg_last_year_months", avg_last_year_months(\
                                               F.col("all_time_mat"),
                                               F.col("Year"),
                                               F.col("DayOfYear")))
    .withColumn("avg_past_month", avg_past_month(\
                                               F.col("all_time_mat"),
                                               F.col("Year"),
                                               F.col("DayOfYear")))
    .withColumn("avg_past_year", avg_past_year(\
                                               F.col("all_time_mat"),
                                               F.col("Year"),
                                               F.col("DayOfYear")))
    .withColumn("all_time_avg", all_time_avg(\
                                               F.col("all_time_mat"),
                                               F.col("Year"),
                                               F.col("DayOfYear"))))


crime_types = tmp.select('y').distinct().rdd.map(lambda r: r[0]).collect()
districts = tmp.select('District').distinct().rdd.map(lambda r: r[0]).collect()

r = tmp.select("DayOfYear","DayOfWeek","Year","Day","District", "all_time_mat",\
                     "Hour","y","avg_last_year_months","avg_past_month","avg_past_year","all_time_avg").collect()

dist_list =  [row["District"] for row in r]
year_list =  [row["Year"] for row in r]
dayofyear_list =  [row["DayOfYear"] for row in r]
hour_list =  [row["Hour"] for row in r]
day_list =  [row["Day"] for row in r]
y_list =  [row["y"] for row in r]
alym_list = [row["avg_last_year_months"] for row in r]
apm_list = [row["avg_past_month"] for row in r]
apy_list = [row["avg_past_year"] for row in r]
ata_list = [row["all_time_avg"] for row in r]
atm_list = [row["all_time_mat"] for row in r]

r = None


def spark_generate_stats(dists, ys, yrs, doys, hrs, alyms, apms, apys, atas, crime_types, districts, years=None):
    if not years:
        years = np.arange(2001,2020)
    leap_years = [2004, 2008, 2012, 2016, 2020]
    res = {}
    for d in districts:
        res[d] = {}
        for t in crime_types:
            res[d][t] = {}
            for year in years:
                if year in leap_years:
                    res[d][t][year] = {day:{} for day in range(1,367)}
                else:
                    res[d][t][year] = {day:{} for day in range(1,366)}
                    
                    
    for dist, y, yr, doy, hr, alym, apm, apy, ata in zip(dists, ys, yrs, doys, hrs, alyms, apms, apys, atas):
        res[dist][y][yr][doy][hr] = {}
        res[dist][y][yr][doy][hr]["alym"] = alym
        res[dist][y][yr][doy][hr]["apm"] = apm
        res[dist][y][yr][doy][hr]["apy"] = apy
        res[dist][y][yr][doy][hr]["ata"] = ata    
    return res

res = spark_generate_stats(dist_list,y_list,year_list,dayofyear_list,hour_list,alym_list,\
                           apm_list,apy_list,ata_list,crime_types, districts)


def get_probability_vectors(district, year, dayofyear, hour, file):
    stats = ["alym","apm","apy","ata"]
    res_vec = []
    for dist, yr, doy, hr in zip(district, year, dayofyear, hour):
        _vec = []
        for s in stats:
            for y in file[dist]:
                try:
                    val = file[dist][y][yr][doy][hr][s]
                    _vec.append(val)
                except: #No statistics available: append 0
                    _vec.append(0)
        res_vec.append(Vectors.dense(_vec))
    return res_vec


probability_vectors = get_probability_vectors(dist_list, year_list, dayofyear_list, hour_list, res)

join_df = sqlContext\
.createDataFrame(zip(day_list, y_list, dist_list, hour_list, atm_list, probability_vectors),
                schema=["jDay","jy","jDistrict", "jHour","all_time_mat", "ProbabilityVector"])

res = None
probability_vectors = None

df = df.join(join_df,\
              ([join_df.jy == df.y,\
                join_df.jDistrict == df.District,
                join_df.jDay == df.Day,
                join_df.jHour == df.Hour]),\
              how='left')\
.drop("jDay","jDistrict","jy","jHour")

join_df.unpersist()

df.toPandas().to_csv('crimes_cleaned_engineered.csv', index=None)
sc.stop()

