package sdml.ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier


import org.apache.spark.sql.SparkSession

/**
 * @Author manshu
 * @Date 2021/8/16 5:53 下午
 * @Version 1.0
 * @Desc
 */
object XgboostPipelineSDAI {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("XgboostPipelineSDAI")
    /*.setMaster("local[*]")*/
    val spark = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()


    val data=spark.sql("select * from  ml.adult")
    //字符类型缺失值填充
    val cols=data.dtypes.map(x=> x._1).filter(t=>t.!=("salary"))
    var df=data.na.replace(cols,Map(""->"NA"))
    for(feature <- cols){
      val indexer = new StringIndexer().setInputCol(feature).setOutputCol(feature.concat("_index"))
      val train_index_model = indexer.fit(df)
      df = train_index_model.transform(df)
      df = df
    }
    //删除原始列
    cols.foreach(s=> df=df.drop(s))

    //特征抽取，进行onehost处理
    val features=df.dtypes.map(x=> x._1).filter(t=>t.!=("salary"))
    val onehot = new OneHotEncoderEstimator().setInputCols(features).setOutputCols(features.map(x=>x.replace("_index","").concat("_onehot")))
    df= onehot.fit(df).transform(df)
    //删除向量索引列
    cols.foreach(s=> df=df.drop(s.concat("_index")))

    //onehost 特征组合
    val onehot_f=df.dtypes.map(x=> x._1).filter(t=>t.!=("salary"))
    val assembler= new VectorAssembler().setInputCols(onehot_f).setOutputCol("features")
    df =assembler.transform(df)

    //切分数据集合
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))
    val labelIndexer=new StringIndexer().setInputCol("salary").setOutputCol("indexedLabel").fit(df)
    val featureIndexer  =  new  VectorIndexer().
      setInputCol( "features" ).
      setOutputCol( "indexedFeatures" ).setMaxCategories(5).fit(df)


    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 5,
      "objective" ->"binary:logistic",
      "num_round" -> 2,
      "silent" -> 1
    ).toMap

    val xgb= new XGBoostClassifier(paramMap).
      setFeaturesCol("indexedFeatures").
      setLabelCol("indexedLabel").
      setMissing(0)
    //组装pipline
    val  pipeline  =  new  Pipeline().setStages(Array(labelIndexer, featureIndexer, xgb))
    val  model  =  pipeline.fit(trainingData)
    val prediction = model.transform(testData)
    val evaluator = new BinaryClassificationEvaluator().
      setLabelCol("indexedLabel").
      setRawPredictionCol("prediction")
    val auc = evaluator.evaluate(prediction)


  }

}
