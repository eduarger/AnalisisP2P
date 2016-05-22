import java.util.Date
import org.apache.log4j.LogManager
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{types, _}
import org.apache.spark.sql.SaveMode
import org.apache.spark._

def cut[A](xs: Seq[A], n: Int) = {
  val m = xs.length
  val targets = (0 to n).map{x => math.round((x.toDouble*m)/n).toInt}
  def snip(xs: Seq[A], ns: Seq[Int], got: Vector[Seq[A]]): Vector[Seq[A]] = {
    if (ns.length<2) got
    else {
      val (i,j) = (ns.head, ns.tail.head)
      snip(xs.drop(j-i), ns.tail, got :+ xs.take(j-i))
    }
  }
  snip(xs, targets, Vector.empty)
}






object analisis {
  def main(args: Array[String]) {
     val logger = LogManager.getLogger("analisis")
     logger.setLevel(Level.INFO)
     logger.setLevel(Level.DEBUG)
     Logger.getLogger("org").setLevel(Level.WARN)
     Logger.getLogger("hive").setLevel(Level.WARN)
     logger.info("Solicitando recursos a Spark")
     val conf = new SparkConf().setAppName("AnalisisP2P")
     val sc = new SparkContext(conf)
     val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
     val numPartitions=50
     val tasaMuestreo=0.1
     val fraudDF="nuevosfraudes"
     val txDF="fraudsidentified"
     logger.info("..........creacion del subcojnunto...............")
     val onlyFrauds = sqlContext.sql("SELECT * FROM "+fraudDF).coalesce(numPartitions)
     logger.info("..........legales...............")
     val onlyLegal = (sqlContext.sql("SELECT * FROM "+txDF +" WHERE fraude ==-1")
     .coalesce(numPartitions).sample(false,tasaMuestreo))
     logger.info("..........no labels...............")
     val onlyNolabel = (sqlContext.sql("SELECT * FROM "+txDF +" WHERE fraude ==0")
     .coalesce(numPartitions).sample(false,tasaMuestreo))
     logger.info("..........Union...............")
     val dfOut= onlyFrauds.unionAll(onlyLegal).unionAll(onlyNolabel)
     dfOut.write.mode(SaveMode.Overwrite).saveAsTable("muestraTx")
    sc.stop()
  }
}
 