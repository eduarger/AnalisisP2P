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
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.ml.PipelineModel
import java.io._

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




def saveRedTable(dataFrameIn: DataFrame, nam: String) : Unit  = {
     val temp=(dataFrameIn
     .drop(col("moneda"))
     .drop(col("tipo_tarjeta"))
     .drop(col("fraudeEx"))
     .drop(col("fraudeSq"))
     .drop(col("noLabel"))
     .drop(col("fecha"))
     .drop(col("metodo_pago"))
     .drop(col("numero_tarjeta"))
     .drop(col("retail_code"))
     .drop(col("documento_cliente"))
     .drop(col("id_sector"))
     .drop(col("id_comercio"))
     .drop(col("banco_emisor"))
     .drop(col("ubicacion"))
     .drop(col("email"))
     .drop(col("categoria_tarjeta"))
     )
     temp.write.mode(SaveMode.Overwrite).saveAsTable(nam)
}




  def main(args: Array[String]) {
     val logger = LogManager.getLogger("analisis")
     logger.setLevel(Level.INFO)
     logger.setLevel(Level.DEBUG)
     Logger.getLogger("org").setLevel(Level.WARN)
     Logger.getLogger("hive").setLevel(Level.WARN)
     logger.info("Solicitando recursos a Spark")
     val conf = new SparkConf().setAppName("AnalisisP2P").set("spark.executor.memory","7373m")
     val sc = new SparkContext(conf)
     val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
     logger.info("........getting the parameters...............")
     // 
     val numPartitions=args(1).toInt
     val numMaxDepth=args(2).toInt
     val arrayNDepth=args.slice(3,3+numMaxDepth).map(_.toInt)
     val numOfTrees= args(3+numMaxDepth).toInt
     val trees=args.slice(3+numMaxDepth+1,3+numMaxDepth+numOfTrees+1).map(_.toInt)
     val numOfBins= args(3+numMaxDepth+numOfTrees+1).toInt
     val arrayBins=args.slice(3+numMaxDepth+numOfTrees+2,3+numMaxDepth+numOfTrees+numOfBins+1).map(_.toInt)
     
     println("Depth: " + arrayNDepth)
     println("trees: " + trees)
     println("Bins: " + arrayBins)
     if (args(0)=="y")
     {
     logger.info("........The Original database will be reading and saving...............")
     // Load and parse the data file, converting it to a DataFrame.
     logger.info("..........reading...............")
     var original = sqlContext.sql("SELECT * FROM datosfinales").coalesce(numPartitions)
     saveRedTable(original,"datosFinalesRed")
     }
     
     logger.info("........Reading datosFinalesRed...............")
     var data = sqlContext.sql("SELECT * FROM datosFinalesRed where fraude==1 OR fraude==-1").coalesce(numPartitions)
     val names = data.columns
     val ignore = Array("idsesion", "fraude", "1_maximo")
     val assembler = (new VectorAssembler()
     .setInputCols( for (i <- names if !(ignore contains i )) yield i)
     .setOutputCol("features"))
     logger.info("........Converting to features...............")
     data = assembler.transform(data)
     //data.write.mode(SaveMode.Overwrite).saveAsTable("labeledDatos")
    // var data = sqlContext.sql("SELECT * FROM labeledDatos where fraude==1 OR fraude==-1").coalesce(numPartitions)
   
     
     logger.info("..........Conviertinedo DF a labeling...............")
     val rows: RDD[Row] = data.rdd
     val labeledPoints: RDD[LabeledPoint]=rows.map(row =>{LabeledPoint(row.getInt(25).toDouble,
     row.getAs[SparseVector](39))})
     import sqlContext.implicits._
     val labeledDF=labeledPoints.toDF().coalesce(numPartitions)
    //labeledDF.write.mode(SaveMode.Overwrite).saveAsTable("labeledDatos")
        
     
     val labelIndexer = (new StringIndexer()
     .setInputCol("label")
     .setOutputCol("indexedLabel")
     .fit(labeledDF))
         
     val featureIndexer = (new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(2)
    .fit(labeledDF))
    
    var textOut="tipo,trees,tp,fn,tn,fp,TPR,SPC,PPV,ACC,F1 \n"
	var textOut2=""
    var textOut3="trees \n"
    
    for (x <- trees) {
    
   
   val nTrees=x  
  
  // for( a <- 1 to k){
   logger.info("..........inicinando ciclo con un valor de trees..............."+ nTrees)
   // Split the data into training and test sets (30% held out for testing)
   val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))

   // Train a RandomForest model.
  val rf = (new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(nTrees))
  

   // Convert indexed labels back to original labels.
  val labelConverter = (new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels))

// Chain indexers and forest in a Pipeline
val pipeline = (new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter)))
// creating the cross validation without paramter grid
// val paramGrid = new ParamGridBuilder().build()
val paramGrid = (new ParamGridBuilder()
  .addGrid(rf.maxDepth, arrayNDepth)
  .addGrid(rf.impurity, Array("entropy", "gini"))
  .addGrid(rf.maxBins, arrayBins)
  .build())
  
val cv = (new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5))

logger.info("..........Training...............")
// Train model.  This also runs the indexers.
val model = cv.fit(trainingData)
// writing the importances
val rfModel = model.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestClassificationModel]
//val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
val importances=rfModel.featureImportances
//val parametros=rfModel.
textOut2=(textOut + nTrees + " " + importances + "\n" )
println(textOut2)


// best parameters
val par = cv.getEstimatorParamMaps.zip(model.avgMetrics)
//saving the best model
textOut3=textOut3 + "---Learned classification forest model" + nTrees+ " ---\n" + par.toString + "\n" + rfModel.toDebugString + "\n\n"




logger.info("..........Testing...............")
// Make predictions.
var predictions = model.transform(testData)

logger.info("..........Calculate Error on test...............")

var predRow: RDD[Row]=predictions.select("label", "predictedLabel").rdd
var predRDD: RDD[(Double, Double)] = (predRow.map(row=>
{(row.getDouble(0), row.getString(1).toDouble)}))
var tp=predRDD.filter(r=>r._1== 1.0 && r._2==1.0).count().toDouble
var fn=predRDD.filter(r=>r._1== 1.0 && r._2== -1.0).count().toDouble
var tn=predRDD.filter(r=>r._1== -1.0 && r._2== -1.0).count().toDouble
var fp=predRDD.filter(r=>r._1== -1.0 && r._2== 1.0).count().toDouble
var TPR = (tp/(tp+fn))*100.0
var SPC = (tn/(fp+tn))*100.0
var PPV= (tp/(tp+fp))*100.0
var acc= ((tp+tn)/(tp+fn+fp+tn))*100.0
var f1= ((2*tp)/(2*tp+fp+fn))*100.0
textOut=(textOut + "test," +  nTrees + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
PPV + "," + acc + "," + f1  + "\n" )
println(textOut)



logger.info("..........Calculate Error on Training...............")
// Make predictions.
predictions = model.transform(trainingData)
predRow = predictions.select("label", "predictedLabel").rdd
predRDD = (predRow.map(row=>
{(row.getDouble(0), row.getString(1).toDouble)}))
tp=predRDD.filter(r=>r._1== 1.0 && r._2==1.0).count().toDouble
fn=predRDD.filter(r=>r._1== 1.0 && r._2== -1.0).count().toDouble
tn=predRDD.filter(r=>r._1== -1.0 && r._2== -1.0).count().toDouble
fp=predRDD.filter(r=>r._1== -1.0 && r._2== 1.0).count().toDouble
TPR = (tp/(tp+fn))*100.0
SPC = (tn/(fp+tn))*100.0
PPV= (tp/(tp+fp))*100.0
acc= ((tp+tn)/(tp+fn+fp+tn))*100.0
f1= ((2*tp)/(2*tp+fp+fn))*100.0
textOut=(textOut + "train," +  nTrees + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
PPV + "," + acc + "," + f1  + "\n" )
println(textOut)

//}

}

logger.info("..........writing the files...............")

val pw = new PrintWriter(new File("Out.txt" ))
pw.write(textOut)
pw.close

val pw2 = new PrintWriter(new File("Out2.txt" ))
pw2.write(textOut2)
pw2.close

val pw3 = new PrintWriter(new File("Out3.txt" ))
pw3.write(textOut3)
pw3.close



// "f1", "precision", "recall", "weightedPrecision", "weightedRecall"







/*Aqui!!!!!!!!!!!!!!!!!!!!!!


spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar n 200 7 5 6 10 30 50 80 100 150
spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar [y=calculo, n=toma tabla]
 [numpartitions] [numero de valores de arboles] [arrya con numero de arboles]
                                                              0 1   2  3 4  5  6  7  8  9 10 11  12 13 14 15
 spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar n 200 3 10 15 25 5 10 25 50 80 100 3  32 64 128 
 
 val numPartitions=args(1).toInt
 val numMaxDepth=args(2).toInt
 val arrayNDepth=args.slice(3,3+numMaxDepth).map(_.toInt)
 val numOfTrees= args(3+numMaxDepth).toInt
 val trees=args.slice(3+numMaxDepth+1,3+numMaxDepth+numOfTrees+1).map(_.toInt)
 val numOfBins= args(3+numMaxDepth+numOfTrees+1).toInt
 val arrayBins=args.slice(3+numMaxDepth+numOfTrees+2,3+numMaxDepth+numOfTrees+numOfBins+1).map(_.toInt)


logger.info("..........Comienza PCA...............")
    val PCs=getPCA(labeledDF,3)
    PCs.write.mode(SaveMode.Overwrite).saveAsTable("PCs")
     logger.info("..........Fin PCA...............")
     
     
     
val evaluator = (new BinaryClassificationEvaluator()
  .setLabelCol("indexedLabel"))
*/ 
     sc.stop()
  }
}


 