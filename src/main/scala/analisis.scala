import java.util.Date
import config.Config
import java.io._
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
  
  val parser = new scopt.OptionParser[Config]("scopt") {
  head("Analisis P2P", "0.1")

  opt[Int]('p', "par").action( (x, c) =>
    c.copy(par = x) ).text("par is an integer of num of partions")
    
  opt[String]('r', "read").action( (x, c) =>
    c.copy(read = x) ).text("read is parameter that says wich is the base table")
  
  opt[String]('o', "out").action( (x, c) =>
    c.copy(out = x) ).text("nameof the outfiles")  
    
  opt[Int]('k', "kfolds").action( (x, c) =>
    c.copy(kfolds = x) ).text("kfolds is an integer of num of folds")

  opt[Seq[Int]]('t', "trees").valueName("<trees1>,<trees1>...").action( (x,c) =>
    c.copy(trees = x) ).text("trees to evaluate")
    
  opt[Seq[String]]('i', "imp").valueName("<impurity>,<impurity>...").action( (x,c) =>
    c.copy(imp = x) ).text("impurity to evaluate")
    
  opt[Seq[Int]]('d', "depth").valueName("<depth1>,<depth2>...").action( (x,c) =>
    c.copy(depth = x) ).text("depth to evaluate")
    
  opt[Seq[Int]]('b', "bins").valueName("<bins1>,<bins2>...").action( (x,c) =>
    c.copy(bins = x) ).text("bins to evaluate")

  help("help").text("prints this usage text")

    
}

// parser.parse returns Option[C]
parser.parse(args, Config()) match {
  case Some(config) =>
    // do stuff 
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
    val numPartitions=config.par
 	val k=config.kfolds
 	val arrayNDepth=config.depth.toArray
 	val trees=config.trees.toArray
 	val arrayBins=config.bins.toArray
 	val opt=config.read
 	val salida=config.out
 	val imp=config.imp.toArray
     
  logger.info("..........buliding grid of parameters...............")
   //val imp= Array("entropy", "gini")
   val grid = for (x <- imp; y <- arrayNDepth; z <- arrayBins) yield(x,y,z)
   for (a <- grid) println(a)
     
     if (opt=="read1")
     {
     logger.info("........The Original database will be reading and saving...............")
     // Load and parse the data file, converting it to a DataFrame.
     logger.info("..........reading...............")
     var original = sqlContext.sql("SELECT * FROM datosfinales").coalesce(numPartitions)
     saveRedTable(original,"datosFinalesRed")
     }
     
      if (opt=="read2")
      {
     logger.info("........Reading datosFinalesRed...............")
     var data = sqlContext.sql("SELECT * FROM datosFinalesRed where fraude==1 OR fraude==-1").coalesce(numPartitions)
     val names = data.columns
     val ignore = Array("idsesion", "fraude", "1_maximo")
     val assembler = (new VectorAssembler()
     .setInputCols( for (i <- names if !(ignore contains i )) yield i)
     .setOutputCol("features"))
     logger.info("........Converting to features...............")
     data = assembler.transform(data)
     data.write.mode(SaveMode.Overwrite).saveAsTable("labeledDatos")
      
      }
    
   
    logger.info("........Reading labeledDatos...............")
     val datos = sqlContext.sql("SELECT * FROM labeledDatos where fraude==1 OR fraude==-1").coalesce(numPartitions)
    
    
     //
    // 
 
     
     logger.info("..........Conviertinedo DF a labeling...............")
     val rows: RDD[Row] = datos.rdd
     val labeledPoints: RDD[LabeledPoint]=rows.map(row =>{LabeledPoint(row.getInt(25).toDouble,
     row.getAs[SparseVector](39))})
     import sqlContext.implicits._
     val labeledDF=labeledPoints.toDF().coalesce(numPartitions)
    //labeledDF.write.mode(SaveMode.Overwrite).saveAsTable("labeledDatos")
        
     logger.info("..........Conviertinedo Features...............")
     
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
    
   
    
    for (params <- grid )
    
    {
    for (x <- trees) {
    
   
   val nTrees=x  
  
   for( a <- 1 to k){
   logger.info("..........inicinando ciclo con un valor de trees..............."+ nTrees)
   println("using(impurity,depth, bins) " + params)
   // Split the data into training and test sets (30% held out for testing)
   val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))

   // Train a RandomForest model.
  val rf = (new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(nTrees)
  .setImpurity(params._1)
  .setMaxDepth(params._2)
  .setMaxBins(params._3)
  )
  

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
/*
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
  */
logger.info("..........Training...............")
// Train model.  This also runs the indexers.
//val model = cv.fit(trainingData)
val model = pipeline.fit(trainingData)
// writing the importances
//val rfModel = model.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestClassificationModel]
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
val importances=rfModel.featureImportances
textOut2=(textOut2 + nTrees + " " + importances + "\n" )



// best parameters
//val par = cv.getEstimatorParamMaps.zip(model.avgMetrics)

//saving the best model
//textOut3=textOut3 + "---Learned classification forest model" + nTrees+ " ---\n" + par.toString + "\n" + rfModel.toDebugString + "\n\n"
textOut3=textOut3 + "---Learned classification forest model" + nTrees+ " ---\n" + params + "\n" + rfModel.toDebugString + "\n\n"



logger.info("..........Testing...............")
// Make predictions.
var predictions = model.transform(testData)

logger.info("..........Calculate Error on test...............")
predictions.cache()
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
PPV + "," + acc + "," + f1  +  "," + params + "\n" )
predictions.unpersist()
println(textOut)



logger.info("..........Calculate Error on Training...............")
// Make predictions.
predictions = model.transform(trainingData)
predictions.cache()
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
PPV + "," + acc + "," + f1 +  "," + params + "\n" )
predictions.unpersist()
println(textOut)

}

}

}

logger.info("..........writing the files...............")

val pw = new PrintWriter(new File(salida+".txt" ))
pw.write(textOut)
pw.close

val pw2 = new PrintWriter(new File(salida+"2.txt" ))
pw2.write(textOut2)
pw2.close

val pw3 = new PrintWriter(new File(salida+"3.txt" ))
pw3.write(textOut3)
pw3.close



  

// "f1", "precision", "recall", "weightedPrecision", "weightedRecall"







/*Aqui!!!!!!!!!!!!!!!!!!!!!!


spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar read 200 7 5 6 10 30 50 80 100 150
spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar [y=calculo, n=toma tabla]
 [numpartitions] [numero de valores de arboles] [arrya con numero de arboles]
                                                              0 1   2  3 4  5  6  7  8  9 10 11  12 13 14 15
 spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar read3 200 5 3 10 15 30 5 10 25 50 80 100 3  32 64 80 
 spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar read3 250 5 1 30 5 10 25 50 80 100 1 64
 
 val numPartitions=args(1).toInt
 val k=args(2).toInt
 val numMaxDepth=args(3).toInt
 val arrayNDepth=args.slice(4,4+numMaxDepth).map(_.toInt)
 val numOfTrees= args(4+numMaxDepth).toInt
 val trees=args.slice(4+numMaxDepth+1,4+numMaxDepth+numOfTrees+1).map(_.toInt)
 val numOfBins= args(4+numMaxDepth+numOfTrees+1).toInt
 val arrayBins=args.slice(4+numMaxDepth+numOfTrees+2,4+numMaxDepth+numOfTrees+numOfBins+1).map(_.toInt)
															  0	    1   2 3 4  5 6  7  8  9  10  11 12
 spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar read3 250 5 1 30 5 10 25 50 80 100 2  64 80


logger.info("..........Comienza PCA...............")
    val PCs=getPCA(labeledDF,3)
    PCs.write.mode(SaveMode.Overwrite).saveAsTable("PCs")
     logger.info("..........Fin PCA...............")
     
     
     
val evaluator = (new BinaryClassificationEvaluator()
  .setLabelCol("indexedLabel"))
*/ 
     sc.stop()
     
     case None =>
    println(".........arguments are bad...............")

}

  }
}


 