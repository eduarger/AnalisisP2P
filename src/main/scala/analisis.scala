import java.util.Date
import config.Config
import java.io._
import org.apache.log4j.LogManager
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SaveMode
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ArrayBuffer


object analisis {


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



def trainMeta(dataFrameIn: DataFrame, parametros: (String, Int, Int),sc: SparkContext, sq: org.apache.spark.sql.SQLContext, trees: Int) : Array(PipelineModel)  = {
   // Split the data into training and test sets (30% held out for testing)
   val logger = LogManager.getLogger("analisis")
   val f=dataFrameIn.where("label=1.0").count().toInt
   val l=dataFrameIn.where("label=-1.0").count().toInt
   val numC=l/f
   val rate = 1.0/(numC)
   val setTosample=dataFrameIn.where("label=-1.0")
   val setFraud=dataFrameIn.where("label=1.0")
   var classificadores = ArrayBuffer[PipelineModel]()
   for( a <- 1 to numC){
   logger.info("..........Subsampling with rate of: " + rate  + "...............")
   val setSampled=setTosample.sample(rate) 
   val setTrain=setSampled.unionAll(setFraud)
   val labelIndexer = (new StringIndexer()
   .setInputCol("label")
   .setOutputCol("indexedLabel")
   .fit(setTrain))
   val featureIndexer = (new VectorIndexer()
   .setInputCol("features")
   .setOutputCol("indexedFeatures")
   .setMaxCategories(2)
   .fit(setTrain))
   val rf = (new RandomForestClassifier()
   .setLabelCol("indexedLabel")
   .setFeaturesCol("indexedFeatures")
   .setNumTrees(trees)
   .setImpurity(parametros._1)
   .setMaxDepth(parametros._2)
   .setMaxBins(parametros._3)
   )  
   // Convert indexed labels back to original labels.
   val labelConverter = (new IndexToString()
   .setInputCol("prediction")
   .setOutputCol("predictedLabel")
   .setLabels(labelIndexer.labels))
   // Chain indexers and forest in a Pipeline
   val pipeline = (new Pipeline()
   .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter)))
   logger.info("..........Training...............")
   var model=pipeline.fit(setTrain)
   classificadores += model
   
  
   }
   return (classificadores)
}
  
def trainWithKmeans(dataFrameIn: DataFrame, parametros: (String, Int, Int),sc: SparkContext, sq: org.apache.spark.sql.SQLContext, trees: Int) : PipelineModel  = {
   // Split the data into training and test sets (30% held out for testing)
   val logger = LogManager.getLogger("analisis")
   val g=dataFrameIn.where("label=1.0").count().toInt
   val setTosample=dataFrameIn.where("label=-1.0").cache()
   val setFraud=dataFrameIn.where("label=1.0")
   logger.info("..........Subsampling with Kmeans...............")
   // Trains a k-means model
   val kmeans = (new KMeans()
   .setK(g)
   .setFeaturesCol("features")
   .setPredictionCol("prediction")
   .setInitMode("random"))      
   val modelK = kmeans.fit(setTosample)

   logger.info("..........Converting centers to new train data set...............")

   val centers=modelK.clusterCenters
   val newValuesRDD: RDD[LabeledPoint]=(sc.parallelize(centers)
   .map(row =>{LabeledPoint(-1.0, row)}))
   import sq.implicits._
   val newValuesDF=newValuesRDD.toDF()
   val setTrain=newValuesDF.unionAll(setFraud)   
   
   val labelIndexer = (new StringIndexer()
   .setInputCol("label")
   .setOutputCol("indexedLabel")
   .fit(setTrain))
   
   
    

   val rf = (new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(trees)
  .setImpurity(parametros._1)
  .setMaxDepth(parametros._2)
  .setMaxBins(parametros._3)
  )
  
  // Convert indexed labels back to original labels.
  val labelConverter = (new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels))

  
 // Chain indexers and forest in a Pipeline
  val pipeline = (new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter)))

  logger.info("..........Training...............")
   val model = pipeline.fit(setTrain)
   
   return (model)
     
     
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
    
  opt[String]('m', "mex").action( (x, c) =>
    c.copy(mex = x) ).text("memory executor (7g or 7000m)")  
  
  opt[String]('h', "hmem").action( (x, c) =>
    c.copy(hmem = x) ).text("memory executor overhead (7g or 7000m)")    
    
  opt[String]('e', "estrategia").action( (x, c) =>
    c.copy(estrategia = x) ).text("strategy to solve the imbalance(kmeans,meta,smote)")    
    
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
 	val memex=config.mex
 	val memover=config.hmem
 	val est=config.estrategia
     
   logger.info("..........buliding grid of parameters...............")
   //val imp= Array("entropy", "gini")
   val grid = for {
           x <- imp
           y <- arrayNDepth
           z <- arrayBins
   } yield(x,y,z)
   
   for (a <- grid) println(a)
    
     logger.info("Solicitando recursos a Spark")
     val conf = new SparkConf().setAppName("AnalisisP2P")
     .set("spark.executor.memory",memex)
     .set("spark.yarn.executor.memoryOverhead", memover)
     val sc = new SparkContext(conf)
     val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
     import sqlContext.implicits._
    
     
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
     logger.info("..........Conviertinedo DF a labeling...............")
     val rows: RDD[Row] = data.rdd
     val labeledPoints: RDD[LabeledPoint]=(rows.map(row =>{LabeledPoint(row.getInt(25).toDouble,
     row.getAs[SparseVector](39))}))
     import sqlContext.implicits._
     val labeledDF=labeledPoints.toDF().coalesce(numPartitions)
        
      }
    
   
    
    logger.info("........Reading labeldf...............")
    val labeledDF = sqlContext.sql("SELECT * FROM labeldf").coalesce(numPartitions)
    
    var textOut="tipo,trees,tp,fn,tn,fp,TPR,SPC,PPV,ACC,F1,MGEO,PEXC,MCC \n"
	var textOut2=""
    var textOut3="trees \n"
    
   
    
    for (params <- grid )
    
    {
    for (x <- trees) {
    
   
   val nTrees=x  
  
   for( a <- 1 to k){
   logger.info("..........inicinando ciclo con un valor de trees..............."+ nTrees)
   println("using(impurity,depth, bins) " + params)
   val Array(trainingData, testData) = labeledDF.cache().randomSplit(Array(0.7, 0.3))
   
   var model = {
   if (est=="kmeans")
   trainWithKmeans(trainingData,params,sc,sqlContext,nTrees)
   else 
   trainWithKmeans(trainingData,params,sc,sqlContext,nTrees)
   }
   
   
   val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
   val importances=rfModel.featureImportances.toArray
   textOut2=(textOut2 + nTrees + "," + importances.mkString(", ") + "\n" )
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
var mGeo=math.sqrt(TPR*SPC)
var pExc=(tp*tn-fn*fp)/((fn+tp)*(tn+fp))
var MCC=(tp*tn-fp*fn)/math.sqrt((fn+tp)*(tn+fp)*(fp+tp)*(fn+tn))
textOut=(textOut + "test," +  nTrees + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
PPV + "," + acc + "," + f1  +  "," +mGeo +  "," + pExc + "," + MCC + "," + params + "\n" )
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
mGeo=math.sqrt(TPR*SPC)
pExc=(tp*tn-fn*fp)/((fn+tp)*(tn+fp))
MCC=(tp*tn-fp*fn)/math.sqrt((fn+tp)*(tn+fp)*(fp+tp)*(fn+tn))
textOut=(textOut + "train," +  nTrees + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
PPV + "," + acc + "," + f1  +  "," +mGeo +  "," + pExc + "," + MCC + "," + params + "\n" )
predictions.unpersist()
println(textOut)

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


}

}

}



     sc.stop()
     
     case None =>
    println(".........arguments are bad...............")

}

  }
}




/*Aqui!!!!!!!!!!!!!!!!!!!!!!



spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -p 500 -h 1000 -m 13500m -r none -o testkm1_ -e kmeans -k 1 -t 50 -i gini,entropy -d 30 -b 72


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
  
  libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"
--repositories "com.github.fommil.netlib" % "all" % "1.1.2"

spark-shell --driver-memory 4g --executor-memory 8g











*/ 


 