package sdml.ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

/**
 * @Author manshu
 * @Date 2021/8/16 5:53 下午
 * @Version 1.0
 * @Desc
 */
object LrPiplineSDAI {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LrPiplineSDAI")
    /*.setMaster("local[*]")*/
    val spark = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()



    val data=spark.sql("select * from  ml.adult")
    val cols=data.dtypes.map(x=> x._1).filter(t=>t.!=("salary"))
    val df=data.na.replace(cols,Map(""->"NA"))

    var Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    for(feature <- cols){

      val indexer = new StringIndexer().setInputCol(feature).setOutputCol(feature.concat("_index"))

      val train_index_model = indexer.fit(trainingData)

      val train_indexed = train_index_model.transform(trainingData)

      val test_indexed = indexer.fit(testData).transform(testData,train_index_model.extractParamMap())

      trainingData = train_indexed

      testData = test_indexed

    }


    val labelIndexer=new StringIndexer().setInputCol("salary").setOutputCol("indexedLabel").fit(trainingData)
    val trainingData_1=labelIndexer.transform(trainingData)
    var features=labelIndexer.transform(trainingData).dtypes.map(x=> x).filter(t=> t._2!=("StringType")).map(x=>x._1)
    var featureIndexer= new VectorAssembler().setInputCols(features).setOutputCol("indexedFeatures")




    //val encoder = new VectorIndexer().setInputCols
    //data.na.replace(,Map(""->"NA"))
    /*val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)*/
    /**
     * LR建模
     * setMaxIter设置最大迭代次数(默认100),具体迭代次数可能在不足最大迭代次数停止(见下一条)
     * setTol设置容错(默认1e-6),每次迭代会计算一个误差,误差值随着迭代次数增加而减小,当误差小于设置容错,则停止迭代
     * setRegParam设置正则化项系数(默认0),正则化主要用于防止过拟合现象,如果数据集较小,特征维数又多,易出现过拟合,考虑增大正则化系数
     * setElasticNetParam正则化范式比(默认0),正则化有两种方式:L1(Lasso)和L2(Ridge),L1用于特征的稀疏化,L2用于防止过拟合
     * setLabelCol设置标签列
     * setFeaturesCol设置特征列
     * setPredictionCol设置预测列
     * setThreshold设置二分类阈值
     */

    val lr = new LogisticRegression().
      setMaxIter(10).
      setRegParam(0.3).
      setElasticNetParam(0).
      setFeaturesCol("indexedFeatures").
      setLabelCol("indexedLabel")
    val pipeline = new Pipeline().setStages(Array(labelIndexer,featureIndexer, lr))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)


    // Select example rows to display.
    predictions.select("indexedLabel","indexedFeatures","probability","prediction").show(5)

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    print(accuracy)

    model.save("/user/fycmb/prediction/model")
    val sameModel = PipelineModel.load("/user/fycmb/prediction/model")

    val predictions_save = sameModel.transform(testData)

    // Select example rows to display.
    predictions_save.select("indexedLabel","indexedFeatures","probability","prediction").show(5)


    val accuracy_save = evaluator.evaluate(predictions_save)

    print(accuracy_save)

  }

}
