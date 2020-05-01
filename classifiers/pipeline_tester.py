from pyspark.sql.types import StringType
from datetime import datetime
import pyspark.sql.functions as F #avoid conflicts with regular python functions
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler 
from pyspark.ml.feature import PCA, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
import numpy as np
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time


class PipelineTester(object):
    """
    PipelineTester contains model specific pipeline testing routines. All conducted experiments are logged into a 
    logging dictionary, which may be used for further evaluation. For each pipeline test you wish to run, specify
    the model you want and which pipeline transformers you wish to test. The rest is taken care of i this class
    """
    
    def __init__(self, data, feauture_cols=None):
        self.logger = {
            "mlp": [],
            "lr": [],
            "dt": []
        }
        self.data = data
        self.fcs = feauture_cols
        
    
    def _top_3_acc(self, predictions):
        _probabilities = [row['probability'] for row in predictions.collect()]
        _trues = [row['target'] for row in predictions.collect()]
        _correct_count = 0
        for _probs, _truth in zip(_probabilities, _trues):
            _top_3 = np.argsort(_probs)[::-1][:3]
            if _truth in _top_3:
                _correct_count +=1
                
        return _correct_count/len(_trues)
            
            
    def _print_error(self, train_acc, test_acc, top_3_train_acc, top_3_test_acc):
        print("Train Error = %g " % (1.0 - train_acc))
        print("Test Error = %g " % (1.0 - test_acc))
        print("Top 3 train Error = %g " % (1.0 - top_3_train_acc))
        print("Top 3 test Error = %g " % (1.0 - top_3_test_acc))
        return
        
        
    def mlp(self, base, train_split = .7, test_split = .3, normalizer = None, scaler = None,
            pca = None, fcs=None, fractions=None):
        start = time.time()
        if fractions:
            _fr = self.data.stat.sampleBy("y", fractions, 42)
            _frac_str = "Yes"
        else:
            _frac_str = "No"
            
        if not fcs:
            fcs = self.fcs.copy()
        
        for i,fc in enumerate(fcs):
            i = i + 1
            if pca:
                if not scaler:
                    print("Add a scaler before applying PCA")
                    return
                _pca = pca
            else:
                _pca = [None]
                
            for k in _pca:
                _start = time.time()
                try:
                    final_output_col = "Features" #if pca, then pcafea..
                    _assembler = VectorAssembler(inputCols= fc , outputCol="Features")
                    _stages = base + [_assembler]
                    if normalizer:
                        _stages = _stages + normalizer
                        final_output_col = "normFeatures"
                    if scaler:
                        _stages = _stages + scaler
                        final_output_col = "scaledFeatures"
                        
                    if k:
                        print("k =",k)
                        _pca_ = PCA(k = k, inputCol = final_output_col, outputCol="PCA_Features")
                        final_output_col = "PCA_Features"
                        _stages = _stages + [_pca_]
                    
                    _pipeline_str = "fc_"+str(i) + ": " + " -> ".join([str(s).split("_")[0] for s in _stages])
                    
                    if 'PCA' in _pipeline_str:
                        _pipeline_str = _pipeline_str.replace("PCA","PCA_"+ str(k)) #add which dimension

                    _pipeline = Pipeline(stages = _stages)
                    
                    if fractions:
                        _pipeline_model = _pipeline.fit(_fr)
                        _piped_df = _pipeline_model.transform(_fr)      
                    else:
                        _pipeline_model = _pipeline.fit(self.data)
                        _piped_df = _pipeline_model.transform(self.data)
                    

                    (_train, _test) = _piped_df.randomSplit([train_split, test_split], seed = 42)

                    output_shape = len(_train.select("target").rdd.flatMap(lambda x: x).distinct().collect())
                    input_shape = len(_train.select(final_output_col).first()[0])

                    layers = [input_shape, 25, output_shape]

                    _mlp = MultilayerPerceptronClassifier(featuresCol = final_output_col,
                                                          labelCol = 'target',
                                                          layers=layers)

                    _mlpModel = _mlp.fit(_train)

                    _test_predictions = _mlpModel.transform(_test)
                    _train_predictions = _mlpModel.transform(_train)

                    _evaluator = MulticlassClassificationEvaluator(
                        predictionCol="prediction",labelCol="target", metricName="accuracy")

                    _test_accuracy = _evaluator.evaluate(_test_predictions)
                    _train_accuracy = _evaluator.evaluate(_train_predictions)
                    
                    try:
                        _top_3_test_acc = self._top_3_acc(_test_predictions)
                        _top_3_train_acc = self._top_3_acc(_train_predictions)
                    except Exception as e:
                        print(e)
                        
                    
                         
                    _ex_time = round((time.time() - _start),1)
                    print("Done with fc_", i, " in ", _ex_time , " seconds")
                    #self._print_error(_train_accuracy, _test_accuracy, _top_3_train_acc, _top_3_test_acc)
                    
                    self.logger["mlp"].append({
                        "time": _ex_time,
                        "pipeline": _pipeline_str,
                        "fractions": _frac_str,
                        "train_acc": _train_accuracy,
                        "test_acc":   _test_accuracy,
                        "top_3_train_acc":  _top_3_train_acc,
                        "top_3_test_acc":  _top_3_test_acc
                    })
                except:
                    print("Failed with fc_",i)
                    continue
        print("MLP PIPELINE TEST: ", round((time.time() - start),1), " seconds")
        return
    
    
    def lr(self, base, train_split = .7, test_split = .3, normalizer=None, scaler = None,
           pca = None, fcs=None, fractions=None):
        start = time.time()
        if fractions:
            _fr = self.data.stat.sampleBy("y", fractions, 42)
            _frac_str = "Yes"
        else:
            _frac_str = "No"
        if not fcs:
            fcs = self.fcs.copy()
        
        for i,fc in enumerate(fcs):
            i = i + 1
            if pca:
                if not scaler:
                    print("Add a scaler before applying PCA")
                    return
                _pca = pca
            else:
                _pca = [None]
                
            for k in _pca:
                _start = time.time()
                try:
                    final_output_col = "Features" #if pca, then pcafea..
                    _assembler = VectorAssembler(inputCols= fc , outputCol="Features")
                    _stages = base + [_assembler]
                    if normalizer:
                        _stages = _stages + normalizer
                        final_output_col = "normFeatures"
                    if scaler:
                        _stages = _stages + scaler
                        final_output_col = "scaledFeatures"
                        
                    if k:
                        print("k =",k)
                        _pca_ = PCA(k = k, inputCol = final_output_col, outputCol="PCA_Features")
                        final_output_col = "PCA_Features"
                        _stages = _stages + [_pca_]
                    
                    _pipeline_str = "fc_"+str(i) + ": " + " -> ".join([str(s).split("_")[0] for s in _stages])
                    
                    if 'PCA' in _pipeline_str:
                        _pipeline_str = _pipeline_str.replace("PCA","PCA_"+ str(k)) #add which dimension

                    _pipeline = Pipeline(stages = _stages)
                    
                    if fractions:
                        _pipeline_model = _pipeline.fit(_fr)
                        _piped_df = _pipeline_model.transform(_fr)      
                    else:
                        _pipeline_model = _pipeline.fit(self.data)
                        _piped_df = _pipeline_model.transform(self.data)
                    

                    (_train, _test) = _piped_df.randomSplit([train_split, test_split], seed = 42)
                    
                    _lr = LogisticRegression(featuresCol = final_output_col,
                                             labelCol = 'target',
                                             maxIter=10)


                    _lrModel = _lr.fit(_train)

                    _test_predictions = _lrModel.transform(_test)
                    _train_predictions = _lrModel.transform(_train)

                    _evaluator = MulticlassClassificationEvaluator(
                        predictionCol="prediction",labelCol="target", metricName="accuracy")

                    _test_accuracy = _evaluator.evaluate(_test_predictions)
                    _train_accuracy = _evaluator.evaluate(_train_predictions)
                    
                    try:
                        _top_3_test_acc = self._top_3_acc(_test_predictions)
                        _top_3_train_acc = self._top_3_acc(_train_predictions)
                    except Exception as e:
                        print(e)
                        
                    
                         
                    _ex_time = round((time.time() - _start),1)
                    print("Done with fc_", i, " in ", _ex_time , " seconds")
                    #self._print_error(_train_accuracy, _test_accuracy, _top_3_train_acc, _top_3_test_acc)
                    
                    self.logger["lr"].append({
                        "time": _ex_time,
                        "pipeline": _pipeline_str,
                        "fractions": _frac_str,
                        "train_acc": _train_accuracy,
                        "test_acc":   _test_accuracy,
                        "top_3_train_acc":  _top_3_train_acc,
                        "top_3_test_acc":  _top_3_test_acc
                    })
                except:
                    print("Failed with fc_",i)
                    continue
        print("Logistic Regression PIPELINE TEST: ", round((time.time() - start),1), " seconds")
        return
    
    
    def dt(self, base, train_split = .7, test_split = .3, normalizer = None, scaler = None, pca = None,
           fcs=None, fractions=None):
        start = time.time()
        if fractions:
            _fr = self.data.stat.sampleBy("y", fractions, 42)
            _frac_str = "Yes"
        else:
            _frac_str = "No"
            
        if not fcs:
            fcs = self.fcs.copy()
        
        for i,fc in enumerate(fcs):
            i = i + 1
            if pca:
                if not scaler:
                    print("Add a scaler before applying PCA")
                    return
                _pca = pca
            else:
                _pca = [None]
                
            for k in _pca:
                _start = time.time()
                try:
                    final_output_col = "Features" #if pca, then pcafea..
                    _assembler = VectorAssembler(inputCols= fc , outputCol="Features")
                    _stages = base + [_assembler]
                    if normalizer:
                        _stages = _stages + normalizer
                        final_output_col = "normFeatures"
                    if scaler:
                        _stages = _stages + scaler
                        final_output_col = "scaledFeatures"
                        
                    if k:
                        print("k =",k)
                        _pca_ = PCA(k = k, inputCol = final_output_col, outputCol="PCA_Features")
                        final_output_col = "PCA_Features"
                        _stages = _stages + [_pca_]
                    
                    _pipeline_str = "fc_"+str(i) + ": " + " -> ".join([str(s).split("_")[0] for s in _stages])
                    
                    if 'PCA' in _pipeline_str:
                        _pipeline_str = _pipeline_str.replace("PCA","PCA_"+ str(k)) #add which dimension

                    _pipeline = Pipeline(stages = _stages)
                    
                    if fractions:
                        _pipeline_model = _pipeline.fit(_fr)
                        _piped_df = _pipeline_model.transform(_fr)      
                    else:
                        _pipeline_model = _pipeline.fit(self.data)
                        _piped_df = _pipeline_model.transform(self.data)
                    

                    (_train, _test) = _piped_df.randomSplit([train_split, test_split], seed = 42)
                    
                    _dt = DecisionTreeClassifier(featuresCol = final_output_col,
                                                 labelCol = 'target')


                    _dtModel = _dt.fit(_train)

                    _test_predictions = _dtModel.transform(_test)
                    _train_predictions = _dtModel.transform(_train)

                    _evaluator = MulticlassClassificationEvaluator(
                        predictionCol="prediction",labelCol="target", metricName="accuracy")

                    _test_accuracy = _evaluator.evaluate(_test_predictions)
                    _train_accuracy = _evaluator.evaluate(_train_predictions)
                    
                    try:
                        _top_3_test_acc = self._top_3_acc(_test_predictions)
                        _top_3_train_acc = self._top_3_acc(_train_predictions)
                    except Exception as e:
                        print(e)
                        
                    
                         
                    _ex_time = round((time.time() - _start),1)
                    print("Done with fc_", i, " in ", _ex_time , " seconds")
                    #self._print_error(_train_accuracy, _test_accuracy, _top_3_train_acc, _top_3_test_acc)
                    
                    self.logger["dt"].append({
                        "time": _ex_time,
                        "pipeline": _pipeline_str,
                        "fractions": _frac_str,
                        "train_acc": _train_accuracy,
                        "test_acc":   _test_accuracy,
                        "top_3_train_acc":  _top_3_train_acc,
                        "top_3_test_acc":  _top_3_test_acc
                    })
                except:
                    print("Failed with fc_",i)
                    continue
        print("Decision Tree PIPELINE TEST: ", round((time.time() - start),1), " seconds")
        return