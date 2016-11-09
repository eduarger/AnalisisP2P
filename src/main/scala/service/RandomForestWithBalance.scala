package service
import org.apache.spark.sql.DataFrame
import org.apache.log4j.LogManager
import org.apache.spark.SparkContext
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{types, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline,PipelineModel}

class RandomForestWithBalance(
    nameTablaToS: String,
    base: DataFrame,
                        colS: String,
                        nameToS:String,
                        p:Double,
                        numP:Int,
                        sqlContext: SQLContext
                      )
 extends Serializable {

   private val lista = scala.collection.mutable.ArrayBuffer.empty[PipelineModel]

 }
