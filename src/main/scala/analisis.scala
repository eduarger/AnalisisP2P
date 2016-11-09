import java.util.Date
import config.Config
import service.DataBase
import java.io._
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors}
import org.apache.spark.mllib.stat.KernelDensity
import scala.collection.mutable.ArrayBuffer



object analisis {

  def proRatio(row:org.apache.spark.sql.Row):Double ={
    val values=row.getAs[DenseVector](0)
    val log = math.log(values(0)/values(1))
    log
  }

  def denCal(sample: RDD[Double], bw:Array[Double], x:Array[Double]):Array[Double] ={
    var densities=Array[Double]()
    for (b <- bw )
    {
      val kd = (new KernelDensity()
      .setSample(sample)
      .setBandwidth(b))
      densities = kd.estimate(x)
  }
    densities
  }
// tiene que entrear un dataframe con la probilidad, similar al rwa predcition
  def getDenText(dataIn:DataFrame,inText:String):String={
    var out=inText+"\n"
    var coefLR: RDD[Double] = dataIn.rdd.map(row=>{proRatio(row)})
    val x=(-20d to 20d by 0.25d).toArray
    val bw= Array(2.0)
    val densidad= denCal(coefLR,bw,x)
    out=densidad.mkString(", ") + "\n"
    out
      }

def trainingRF(dataFrameIn: DataFrame,
    parametros: (String, Int, Int),
    sc: SparkContext,
    sq: org.apache.spark.sql.SQLContext,
    trees: Int,
    laIndx : StringIndexerModel,
    feIndx: VectorIndexerModel,
    laConv : IndexToString ) : PipelineModel  = {

  val logger = LogManager.getLogger("analisis")
  logger.info("..........training RF...............")
  val rf = (new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(trees)
  .setImpurity(parametros._1)
  .setMaxDepth(parametros._2)
  .setMaxBins(parametros._3)
  )
   // Chain indexers and forest in a Pipeline
  val pipeline = (new Pipeline()
  .setStages(Array(laIndx, feIndx, rf, laConv)))
  logger.info("..........Training...............")
  val model = pipeline.fit(dataFrameIn)
  model

     }


def main(args: Array[String]) {

  val parser = new scopt.OptionParser[Config]("scopt") {
  head("Analisis P2P", "0.1")
  opt[String]('i', "in").action( (x, c) =>
  c.copy(in = x) ).text("base table")
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
     val logger = LogManager.getLogger("analisis")
     logger.setLevel(Level.INFO)
     logger.setLevel(Level.DEBUG)
     Logger.getLogger("org").setLevel(Level.WARN)
     Logger.getLogger("hive").setLevel(Level.WARN)
     logger.info("........getting the parameters...............")
     //
     val tablaBase=config.in
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
     // printing the grid
     logger.info("..........Here the grid constructed...............")
     for (a <- grid) println(a)
     //Begin the analysis
     logger.info("Solicitando recursos a Spark")
     val conf = new SparkConf().setAppName("AnalisisP2P")
     .set("spark.executor.memory",memex)
     .set("spark.yarn.executor.memoryOverhead", memover)
     val sc = new SparkContext(conf)
     val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
     import sqlContext.implicits._
     // read the tabla base
     val db=new DataBase(tablaBase,numPartitions,sqlContext)
     // if opt==0 the table is readed otherwise the dataframe
     // is calculated from zero
     val labeledDF = db.getDataFrameLabeledLegalFraud(opt=="0").cache()
     var textOut="tipo,trees,tp,fn,tn,fp,TPR,SPC,PPV,ACC,F1,MGEO,PEXC,MCC \n"
     var textOut2=""
     var textOut3="trees \n"
     var txtDendsidad=""

     val labelIndexer = (new StringIndexer()
     .setInputCol("label")
     .setOutputCol("indexedLabel")
     .fit(labeledDF))
     val featureIndexer = (new VectorIndexer()
     .setInputCol("features")
     .setOutputCol("indexedFeatures")
     .setMaxCategories(2)
     .fit(labeledDF))
     // Convert indexed labels back to original labels.
     val labelConverter = (new IndexToString()
     .setInputCol("prediction")
     .setOutputCol("predictedLabel")
     .setLabels(labelIndexer.labels))


    for (params <- grid )

    {
     for (x <- trees) {


   val nTrees=x

   for( a <- 1 to k){
   logger.info("..........inicinando ciclo con un valor de trees..............."+ nTrees)
   println("using(impurity,depth, bins) " + params)
   val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))


   var model = {
   if (est=="rfpure")
     trainingRF(trainingData,params,sc,sqlContext,nTrees,labelIndexer,featureIndexer,labelConverter)
   if(est=="metaclasificador")
     trainingRF(trainingData,params,sc,sqlContext,nTrees,labelIndexer,featureIndexer,labelConverter)
   else
     trainingRF(trainingData,params,sc,sqlContext,nTrees,labelIndexer,featureIndexer,labelConverter)
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
   var predRDD: RDD[(Double, Double)] = (predRow.map(row=>{(row.getDouble(0), row.getString(1).toDouble)}))
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
  predRDD = (predRow.map(row=>{(row.getDouble(0), row.getString(1).toDouble)}))
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
  val pw = new PrintWriter(new File(salida+"Confusion.txt" ))
  pw.write(textOut)
  pw.close
  val pw2 = new PrintWriter(new File(salida+"Importances.txt" ))
  pw2.write(textOut2)
  pw2.close
  val pw3 = new PrintWriter(new File(salida+"Model.txt" ))
  pw3.write(textOut3)
  pw3.close
  logger.info("..........getting densidity...............")
  val legal=trainingData.where("label=-1.0")
  val predLegal = model.transform(legal)
  var predDen = predLegal.select("rawPrediction")
  logger.info("..........getting densidity legal...............")
  val d1= getDenText(predDen,"Legal,"+nTrees+ ","+params)
  val fraude=trainingData.where("label=1.0")
  val predFraud= model.transform(fraude)
  predDen = predFraud.select("rawPrediction")
  logger.info("..........getting densidity fraude...............")
  val d2= getDenText(predDen,"Fraude,"+nTrees+ ","+params)
  txtDendsidad=d1+d2
  val pwdensidad = new PrintWriter(new File(salida+"_denisad.txt" ))

  pw.write(txtDendsidad)
  pw.close

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



spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i base_tarjeta -p 500 -h 1000 -m 13500m -r 1 -o testrf1 -e rfpure -k 1 -t 50 -i gini,entropy -d 30 -b 72

spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i base_tarjeta -p 100 -h 1000 -m 13500m -r 1 -o testrf1 -e rfpure -k 1 -t 50 -i gini -d 30 -b 32
spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i base_tarjeta_complete -p 100 -h 1000 -m 13500m -r 1 -o testrf1 -e rfpure -k 1 -t 25 -i gini -d 30 -b 32




opt[String]('i', "in").action( (x, c) =>
c.copy(in = x) ).text("base table")

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
