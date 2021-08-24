package sdml.ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession

/**
 * @Author manshu
 * @Date 2021/8/16 5:53 下午
 * @Version 1.0
 * @Desc
 */
object LrModelSDAI {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LrModelSDAI")
    /*.setMaster("local[*]")*/
    val spark = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()



    val data=spark.sql("select * from  ml.adult")
    //data.na.fill("NA")
    val cols=data.dtypes.map(x=> x._1).filter(t=>t.!=("salary"))
    val df=data.na.replace(cols,Map(""->"NA"))

    var Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    for(feature <- cols){

      val indexer = new StringIndexer()

        .setInputCol(feature)

        .setOutputCol(feature.concat("_index"))

      val train_index_model = indexer.fit(trainingData)

      val train_indexed = train_index_model.transform(trainingData)

      val test_indexed = indexer.fit(testData).transform(testData,train_index_model.extractParamMap())

      trainingData = train_indexed

      testData = test_indexed

    }

    val trainingData_1 = new StringIndexer().
      setInputCol("salary").
      setOutputCol("label").
      fit(trainingData).
      transform(trainingData)
    val testData_1 = new StringIndexer().
      setInputCol("salary").
      setOutputCol("label").
      fit(testData).
      transform(testData)

    var features=trainingData_1.dtypes.map(x=> x).filter(t=> t._2!=("StringType")).map(x=>x._1)

    var trainingData_2= new VectorAssembler().
      setInputCols(features).
      setOutputCol("features").
      transform(trainingData_1)

    var testData_2= new VectorAssembler().
      setInputCols(features).
      setOutputCol("features").
      transform(testData_1)
    val sameModel = PipelineModel.load("/user/fycmb/ml/lr")

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
      setFeaturesCol("features").
      setLabelCol("label").
      setPredictionCol("predict")
    val lrModel = lr.fit(trainingData_2)
    val pipeline = new Pipeline().setStages(Array(trainingData_2.select("label"), trainingData_2.select("features"), lr))


    val predictions = lrModel.transform(testData_2)
    predictions.select("label", "predict", "features").show(5)
    println(s"每个特征对应系数: ${lrModel.coefficients} 截距: ${lrModel.intercept}")
    //auc
    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("label").
      setPredictionCol("predict").
      setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    print(accuracy)

    //pre 加权准确
    val evaluator_pre = new MulticlassClassificationEvaluator().
      setLabelCol("label").
      setPredictionCol("predict").
      setMetricName("weightedPrecision")
    val pre = evaluator_pre.evaluate(predictions)

    //召回
    val evaluator_recall = new MulticlassClassificationEvaluator().
      setLabelCol("label").
      setPredictionCol("predict").
      setMetricName("weightedRecall")
    val recall = evaluator_recall.evaluate(predictions)

    //F1
    val evaluator_f1 = new MulticlassClassificationEvaluator().
      setLabelCol("label").
      setPredictionCol("predict").
      setMetricName("f1")
    val f1 = evaluator_f1.evaluate(predictions)

    //显示每次迭代的时候的目标值，即损失值+正则项
    val trainingSummary=lrModel.summary
    val objectiveHistory=trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss=>println(loss))

    val binarySummary=trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
    //评估指标
    //roc值
    val roc=binarySummary.roc
    roc.show(false)
    val AUC=binarySummary.areaUnderROC
    println(s"areaUnderRoc:${AUC}")


  }

}
