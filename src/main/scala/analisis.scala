import java.util.Date
import org.apache.log4j.LogManager
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{types, _}
import org.apache.spark.sql.SaveMode
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame


object analisis {

 def getPCA(dataFrame: DataFrame, nc: Int): DataFrame = {
    val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(nc)
    .fit(dataFrame)
    val pcaDF = pca.transform(dataFrame)
    return pcaDF
        
  }


def featureAnalyzer(dataFrameIn: DataFrame, ns: Int,np: Int, nc: Int, label: String): DataFrame = {
//dataFrameIn: a dataframe of the feature set to analyze
//ns: number of samples to do the Bagging
//np: number of partitions
//nc: number of classes
// first we calculate the sampling rate
     val sampling=(1.0)/ns.toDouble
     val onlyNolabel = (sqlContext.sql("SELECT * FROM "+txDF +" WHERE fraude ==0")
     .coalesce(numPartitions).sample(true,tasaMuestreo))
    val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(nc)
    .fit(dataFrame)
    val pcaDF = pca.transform(dataFrame)
    return pcaDF }    






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
     /*val tasaMuestreo=0.1
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
     
     val avgByEmail = (sqlContext.sql("SELECT * FROM  muestraemail").coalesce(numPartitions)
     .select(col("idsesion").alias("idemail"), col("6horas").alias("6hemail"), col("24horas").alias("24hemail"),
      col("15dias").alias("15demail"), col("30dias").alias("30demail"),
      col("60dias").alias("60demail"), col("365dias").alias("365demail")))
     val avgByCard = (sqlContext.sql("SELECT * FROM  muestratarjeta").coalesce(numPartitions)
     .select(col("idsesion").alias("idcard"), col("6horas").alias("6hcard"), col("24horas").alias("24hcard"),
      col("15dias").alias("15dcard"), col("30dias").alias("30dcard"),
      col("60dias").alias("60dcard"), col("365dias").alias("365dcard")))
     var joinDF=avgByEmail.join(avgByCard,avgByEmail("idemail")===avgByCard("idcard")).drop(col("idemail"))
      logger.info("..........Fin Join card y email ...............")
     joinDF.write.mode(SaveMode.Overwrite).saveAsTable("muestraTxAll")
     val avgAll = sqlContext.sql("SELECT * FROM  muestraTxAll").coalesce(numPartitions)
     val freqEmail=sqlContext.sql("SELECT idsesion, probabilidad_ubicacion_email FROM  muestrafreqemail").coalesce(numPartitions)
     joinDF=avgAll.join(freqEmail,avgAll("idcard")===freqEmail("idsesion")).drop(col("idsesion"))
     joinDF.write.mode(SaveMode.Overwrite).saveAsTable("muestraTxAll1")
     logger.info("..........Fin Join freqemail...............")
     val avgAndFreq = sqlContext.sql("SELECT * FROM  muestraTxAll1").coalesce(numPartitions)
     val freqCard=sqlContext.sql("SELECT idsesion, probabilidad_ubicacion_numero_tarjeta FROM  muestrafreqtar").coalesce(numPartitions)
     joinDF=avgAndFreq.join(freqCard,avgAndFreq("idcard")===freqCard("idsesion")).drop(col("idsesion"))
     joinDF.write.mode(SaveMode.Overwrite).saveAsTable("muestraTxAll2")
     logger.info("..........Fin Join freqCard...............")
     val table = (sqlContext.sql("SELECT idsesion, tipo_tarjeta, monto, moneda, valida_cifin, fraude FROM  fraudsidentified")
     .coalesce(numPartitions))
     val tableAll = sqlContext.sql("SELECT * FROM  muestraTxAll2").coalesce(numPartitions)
     joinDF=tableAll.join(table,tableAll("idcard")===table("idsesion")).drop(col("idcard"))     
     joinDF.write.mode(SaveMode.Overwrite).saveAsTable("muestraTxAllF")
     logger.info("..........Fin del join final...............")
	  */

// Load and parse the data file, converting it to a DataFrame.
     val original = sqlContext.sql("SELECT * FROM muestratxallf").coalesce(numPartitions)
     val data=(original
     .drop(col("idsesion"))
     .drop(col("moneda"))
     .drop(col("tipo_tarjeta")))
     
     logger.info("..........Conviertinedo DF a labeling...............")
     val rows: RDD[Row] = data.rdd
     val labeledPoints: RDD[LabeledPoint]=rows.map(row =>{LabeledPoint(row.getInt(16).toDouble,
     Vectors.dense(row.getDouble(0), row.getDouble(1),row.getDouble(2), row.getDouble(3),
     row.getDouble(4), row.getDouble(5),row.getDouble(6), row.getDouble(7),
     row.getDouble(8), row.getDouble(9),row.getDouble(10), row.getDouble(11),
     row.getDouble(12), row.getDouble(13),row.getDouble(14), row.getByte(15).toDouble))
     })
    import sqlContext.implicits._
    val labeledDF=labeledPoints.toDF()
    logger.info("..........Comienza PCA...............")
    val PCs=getPCA(labeledDF,3)
    PCs.write.mode(SaveMode.Overwrite).saveAsTable("PCs")
     logger.info("..........Fin PCA...............")
     
     
     val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(labeledDF)
     
   
    
     val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(labeledDF)
  
  
// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(100)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
logger.info("..........Training...............")
// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData)

logger.info("..........Testing...............")
// Make predictions.
val predictions = model.transform(testData)



logger.info("..........Calculate Error...............")
val evaluator = (new BinaryClassificationEvaluator()
  .setLabelCol("indexedLabel"))

val area = evaluator.evaluate(predictions)
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]


println(area)
println(rfModel.featureImportances)


// "f1", "precision", "recall", "weightedPrecision", "weightedRecall"

//predictions.select("predictedLabel", "label", "features").show(10)

val onlyF= labeledDF.filter("label= 1")
val N=onlyF.count()
val nF = model.transform(onlyF).filter("predictedLabel=1").count()
val rd=nF/N.toDouble
println(N)
println(rd)




/*

*/ 
     sc.stop()
  }
}


 