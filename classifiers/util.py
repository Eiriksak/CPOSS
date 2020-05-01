import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.linalg import Vectors, MatrixUDT, VectorUDT, DenseMatrix, DenseVector
from pyspark.ml.pipeline import Transformer
import math
from pyspark.sql.types import *
import numpy as np

"""
Define classes which inherits the Spark transformer properties.
These classes can be used in MLlib pipelines
These techniques are inspired by:
https://towardsdatascience.com/pyspark-wrap-your-feature-engineering-in-a-pipeline-ee63bdb913

Date derivatives transformers:
1. Assign unique ID in this() so the object is immutable and identifiable within
   the pipeline itself.
2. Try to make a new instance with the same unique ID in copy(). It will copy
   the embedded and extra params over and return a new instance.
3. Allways check wheter the input field is in the correct format in input_check()
4. _transform() performes the transformation itself

Example usage:
get_month = GetMonth(inputCol="Day")
fp = Pipeline(stages = [get_month])
fpfit = fp.fit(df)
df = fpfit.transform(df)

User defined functions for feauture engineering is also available in this util package
"""

class GetWeekOfYear(Transformer):
    """
    inputCol: DateType
    """
    def __init__(self, inputCol, outputCol='WeekOfYear'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getWeekOfYear"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.DateType()):
            raise Exception('Input type %s is not supported for WeekOfYear. Convert to DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.weekofyear(df[self.inputCol]))




class GetDayOfWeek(Transformer):
    """
    inputCol: DateType
    """
    def __init__(self, inputCol, outputCol='DayOfWeek'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getDayOfWeek"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.DateType()):
            raise Exception('Input type %s is not supported for DayOfWeek. Convert to DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.dayofweek(df[self.inputCol]))



class GetDayOfYear(Transformer):
    """
    inputCol: DateType
    """
    def __init__(self, inputCol, outputCol='DayOfYear'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getDayOfYear"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.DateType()):
            raise Exception('Input type %s is not supported for DayOfYear. Convert to DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.dayofyear(df[self.inputCol]))



class GetDayOfMonth(Transformer):
    """
    inputCol: DateType
    """
    def __init__(self, inputCol, outputCol='DayOfMonth'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getDayOfMonth"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.DateType()):
            raise Exception('Input type %s is not supported for DayOfMonth. Convert to DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.dayofmonth(df[self.inputCol]))


class GetQuarter(Transformer):
    """
    inputCol: Timestpamps
    """
    def __init__(self, inputCol, outputCol='Quarter'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getQuarter"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.TimestampType()):
            raise Exception('Input type %s is not supported for GetQuerter. Convert to Timestamps' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.quarter(df[self.inputCol]))



class GetMinute(Transformer):
    """
    inputCol: Timestpamps
    """
    def __init__(self, inputCol, outputCol='Minute'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getMinute"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.TimestampType()):
            raise Exception('Input type %s is not supported for GetMinute. Convert to Timestamps' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.minute(df[self.inputCol]))



class GetHour(Transformer):
    """
    inputCol: Timestpamps
    """
    def __init__(self, inputCol, outputCol='Hour'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getHour"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.TimestampType()):
            raise Exception('Input type %s is not supported for GetHour. Convert to Timestamps' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.hour(df[self.inputCol]))




class GetMonth(Transformer):
    """
    inputCol: DateType
    """
    def __init__(self, inputCol, outputCol='Month'):
        self.inputCol = inputCol 
        self.outputCol = outputCol 

    def this():
        this(Identifiable.randomUID("getMonth"))

    def copy(extra):
        defaultCopy(extra)

    def input_check(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != T.DateType()):
            raise Exception('Input type %s is not supported for GetMonth. Convert to DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.month(df[self.inputCol]))




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