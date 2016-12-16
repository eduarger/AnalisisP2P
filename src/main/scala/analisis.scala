import java.util.Date
import config.Config
import service.DataBase
import service.Metrics
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

  def proRatio(row:org.apache.spark.sql.Row, inv:Boolean):Double ={
    val values=row.getAs[DenseVector](0)
    val small=0.00001
    val log = {
      if(inv&&values(1)!=0&&values(0)!=0.0)
        math.log(values(1)/values(0))
      else if(inv&&values(1)==0&&values(0)!=0.0)
        math.log(small/values(0))
      else if(inv&&values(1)!=0&&values(0)==0.0)
          math.log(values(1)/small)
      else if(!inv&&values(1)!=0&&values(0)!=0.0)
        math.log(values(0)/values(1))
      else if(!inv&&values(1)==0&&values(0)!=0.0)
        math.log(values(0)/small)
      else if(!inv&&values(1)!=0&&values(0)==0.0)
        math.log(small/values(1))
      else
        math.log(values(1)/values(0))
    }
    log
  }

  def denCal(sample: RDD[Double], bw:Double, x:Array[Double]):Array[Double] ={
    var densities=Array[Double]()
      val kd = (new KernelDensity()
      .setSample(sample)
      .setBandwidth(bw))
      densities = kd.estimate(x)
      densities
  }

  // tiene que entrear un dataframe con la probilidad, similar al rwa predcition
  def getDenText(dataIn:DataFrame,inText:String,ejeX:Array[Double],inv:Boolean):String={
    var out=inText+","
    var coefLR: RDD[Double] = dataIn.rdd.map(row=>{proRatio(row,inv)})
    val x=ejeX
    val n=coefLR.count.toDouble
    val h=coefLR.stdev*scala.math.pow((4.0/3.0/n),1.0/5.0)
    val bw=0.1
    val densidad= denCal(coefLR,bw,x)
    out=out+densidad.mkString(", ") + "\n"
    out
      }

// parametros(trees,impurity,depth, bins)
def trainingRF(dataFrameIn: DataFrame,
    parametros: (Int,String, Int, Int),
    sc: SparkContext,
    sq: org.apache.spark.sql.SQLContext,
    laIndx : StringIndexerModel,
    feIndx: VectorIndexerModel,
    laConv : IndexToString ) : PipelineModel  = {

  @transient val logger = LogManager.getLogger("analisis")
  logger.info("..........training RF...............")
  val rf = (new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(parametros._1)
  .setImpurity(parametros._2)
  .setMaxDepth(parametros._3)
  .setMaxBins(parametros._4)
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
  opt[Seq[Int]]('a', "axis").valueName("<start>,<end>").action( (x,c) =>
  c.copy(axes = x) ).text("range of axis of the densidity")
  opt[String]('f', "filter").action( (x, c) =>
  c.copy(filter = x) ).text("filters of the tabla of input")
  help("help").text("prints this usage text")

}

// parser.parse returns Option[C]
  parser.parse(args, Config()) match {
  case Some(config) =>
     val logger= LogManager.getLogger("analisis")
     logger.setLevel(Level.INFO)
     logger.setLevel(Level.DEBUG)
     Logger.getLogger("org").setLevel(Level.WARN)
     Logger.getLogger("hive").setLevel(Level.WARN)
     logger.info("........getting the parameters...............")
     //lectura de parametros
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
     val filtros=config.filter
     val ejesX=config.axes.toArray
     logger.info("Taking the folliwng filters: "+ filtros)
     logger.info("..........buliding grid of parameters...............")
     val grid = for {
           s <- trees
           x <- imp
           y <- arrayNDepth
           z <- arrayBins
     } yield(s,x,y,z)
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
     val db=new DataBase(tablaBase,numPartitions,sqlContext,filtros)
     // if opt==0 the table is readed otherwise the dataframe
     // is calculated from zero
     val labeledDF = db.getDataFrameLabeledLegalFraud(opt=="0").cache()
     var textOut="tipo,trees,tp,fn,tn,fp,SENS,SPC,PRE,ACC,F1,MGEO,PEXC,MCC \n"
     var textOut2=""
     var textOut3="trees \n"
     var txtDendsidad=""
     var txtDendsidadAc=""

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


     for ( a <- 1 to k)
     {
      for(params <- grid){
   logger.info("..........inicinando ciclo con un valor de trees..............."+ params._1)
   println("using(trees,impurity,depth, bins) " + params)
   val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))


   var model = {
   if (est=="rfpure")
     trainingRF(trainingData,params,sc,sqlContext,labelIndexer,featureIndexer,labelConverter)
   if(est=="metaclasificador")
     trainingRF(trainingData,params,sc,sqlContext,labelIndexer,featureIndexer,labelConverter)
   else
     trainingRF(trainingData,params,sc,sqlContext,labelIndexer,featureIndexer,labelConverter)
   }
   val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
   val importances=rfModel.featureImportances.toArray
   textOut2=(textOut2 + params._1 + "," + importances.mkString(", ") + "\n" )
   textOut3=textOut3 + "---Learned classification forest model" + params._1+ " ---\n" + params + "\n" + rfModel.toDebugString + "\n\n"
   logger.info("..........Testing...............")
   // Make predictions.
   var predictions = model.transform(testData)
   logger.info("..........Calculate Error on test...............")
   predictions.cache()
   val testMetrics=new Metrics(predictions, Array(1.0,-1.0))
   var tp=testMetrics.tp
   var fn=testMetrics.fn
   var tn=testMetrics.tn
   var fp=testMetrics.fp
   var TPR = testMetrics.sens
   var SPC = testMetrics.spc
   var PPV= testMetrics.pre
   var acc= testMetrics.acc
   var f1= testMetrics.f1
   var mGeo=testMetrics.mGeo
   var pExc=testMetrics.pExc
   var MCC=testMetrics.MCC
   textOut=(textOut + "test," +  params._1 + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
      PPV + "," + acc + "," + f1  +  "," +mGeo +  "," + pExc + "," + MCC + "," + params + "\n" )
   predictions.unpersist()
  // println(textOut)

  logger.info("..........Calculate Error on Training...............")
  // Make predictions.
  predictions = model.transform(trainingData)
  predictions.cache()
  val trainMetrics=new Metrics(predictions, Array(1.0,-1.0))
  tp=trainMetrics.tp
  fn=trainMetrics.fn
  tn=trainMetrics.tn
  fp=trainMetrics.fp
  TPR = trainMetrics.sens
  SPC = trainMetrics.spc
  PPV= trainMetrics.pre
  acc= trainMetrics.acc
  f1= trainMetrics.f1
  mGeo=trainMetrics.mGeo
  pExc=trainMetrics.pExc
  MCC=trainMetrics.MCC
  textOut=(textOut + "train," +  params._1 + ","+ tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
  PPV + "," + acc + "," + f1  +  "," +mGeo +  "," + pExc + "," + MCC + "," + params + "\n" )
  //println(textOut)
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
  // define the x axis
  val axis=(ejesX(0).toDouble to ejesX(1).toDouble by 0.5d).toArray
  logger.info("..........getting densidity...............")
  val predLegal=predictions.where("label=-1.0")
  var predDen = predLegal.select("probability")
  logger.info("..........getting densidity legal...............")
  val d1= getDenText(predDen,"Legal,"+params._1+ ","+params,axis,true)
  val predFraud=predictions.where("label=1.0")
  predDen = predFraud.select("probability")
  logger.info("..........getting densidity fraude...............")
  val d2= getDenText(predDen,"Fraude,"+params._1+ ","+params,axis,true)
  txtDendsidad="clase,trees,imp,depth,bines,"+axis.mkString(", ") + "\n"+d1+d2
  txtDendsidadAc=txtDendsidadAc+ "\n"+ txtDendsidad
  val pwdensidad = new PrintWriter(new File(salida+"_denisad.csv" ))
  pwdensidad.write(txtDendsidadAc)
  pwdensidad.close
  predictions.unpersist()
  logger.info("..........termino..............")
}

 //

}
logger.info("..........termino programa..............")
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

spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 13500m -r 1 -o rf1 -e rfpure -k 5 -t 1,10,25,50,100 -i gini,entropy -d 10,20,32 -b 32,72

spark-submit --driver-memory 4g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 13500m -r 1 -o test1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72


spark-submit --master yarn --deploy-mode cluster --driver-memory 1g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 1 -o p1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72

spark-submit --driver-memory 1g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 1 -o p1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72

nohup spark-submit a.py > archivo_salida 2>&1&
nohup spark-submit --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 1 -o p1 -e rfpure -k 4 -t 1,10,25,50,100 -i gini,entropy -d 10,20,30 -b 32,72 -a -30,30 > archivo_salida 2>&1&


nohup spark-submit --driver-memory 6g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 10000m -r 0 -o p1 -e rfpure -k 4 -t 25,50,100 -i gini,entropy -d 10,20,30 -b 32,72 -a -25,25 > archivo_salida 2>&1&
nohup spark-submit --driver-memory 10g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 12000m -r 0 -o p1 -e rfpure -k 4 -t 100 -i gini,entropy -d 10,20,30 -b 32,72 -a -25,25 > archivo_salida 2>&1&

nohup spark-submit --num-executors 2 --driver-memory 10g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 12000m -r 0 -o p1 -e rfpure -k 4 -t 2 -i gini,entropy -d 10,20,30 -b 32,72 -a -25,25 > archivo_salida 2>&1&

nohup spark-submit --num-executors 3 --driver-memory 10g --class "analisis" AnalisisP2P-assembly-1.0.jar -i variables_finales_tarjeta -p 200 -h 1000 -m 12000m -r 0 -o pbinhigh -e rfpure -k 4 -t 1,2,25,100 -i gini,entropy -d 30 -b 128,384 -a -25,25 > archivo_salida 2>&1&



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
