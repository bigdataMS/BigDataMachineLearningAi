package sdml.ml

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * @Author manshu
 * @Date 2020/8/16 5:53 下午
 * @Version 1.0
 * @Desc
 */
object AlsPipelineSDAI {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("AlsPipelineSDAI")
    /*.setMaster("local[*]")*/
    val spark = SparkSession.builder().config(conf).enableHiveSupport().getOrCreate()




  }

}
