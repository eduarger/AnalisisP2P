package service
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.clustering.KMeans
//import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors}
import org.apache.spark.mllib.stat.KernelDensity
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{types, _}
import org.apache.spark.ml.{Pipeline,PipelineModel}

/**
 * Clase para que define un clasificador basado en Random Forest
 * Cada random forest tendra un solo arbol pero sera entrenado
 * como un subconjunto balanceado
 * Para iniciar tiene los siguentes parametros:
 * baseTrain: dataframe con labeledpoint
 * parameters=(trees,impurity,depth, bins)
 * numP: numero de particiones
 * sqlContext: contexto de SQL, sc: spark context
 * laIndx: feIndx:  laConv:
 *
*/
class RandomForestWithBalance(
    baseTrain: DataFrame,
    parameters: (Int,String, Int, Int),
    numP: Int,
    laIndx : StringIndexerModel,
    feIndx: VectorIndexerModel,
    laConv : IndexToString,
    labelMinor: Double,
    sqlContext: SQLContext,
    sc: SparkContext)
    extends Serializable {

   val numPartitions=numP
   private var metaClassifier = scala.collection.mutable.ArrayBuffer.empty[RandomForestModel]//[PipelineModel]
   // training dataset empty
   private var trainDataset=sqlContext.createDataFrame(sc.emptyRDD[Row], baseTrain.schema)
   // relation for balance the training datasets
   val numOfMinorClass= baseTrain.where("label="+labelMinor).count()
   val numOfMayorClass= baseTrain.where("label!="+labelMinor).count()
   // calculate the smaple fraction and the number of classifiers
   private var sampleFraction=numOfMinorClass.toDouble/numOfMayorClass.toDouble
   private val fractionInv=numOfMayorClass/numOfMinorClass
   // avoid odd values
   private var numClassifiers=if (fractionInv%2==0) fractionInv + 1 else fractionInv
   // indexers and converters
   private val labelIndex=laIndx
   private val featureIndex=feIndx
   private val labelConvert=laConv
   @transient private val logger = LogManager.getLogger("RandomForestWithBalance")
   logger.info("Number of classifiers(trees): "+numClassifiers)
   logger.info("sample rate is: "+sampleFraction)

// set the training dataset
   def getTrainingBalancedSet() : Unit ={
     val dfMinor= baseTrain.where("label="+labelMinor)
     val dfMayor= baseTrain.where("label!="+labelMinor).sample(false,sampleFraction)
     trainDataset=dfMinor.unionAll(dfMayor)
   }
// traing one tree
   def trainingTree() : RandomForestModel  = {
   //PipelineModel  = {
     // set the training dataset in each traiing
     getTrainingBalancedSet()
val categoricalFeaturesInfo=featureIndexer.categoryMaps
     val categoricalFeaturesInfo = Map[Int, Int]((1,3),(2,5))/// TODO
     val model = RandomForest.trainClassifier(trainDataset, numClasses, categoricalFeaturesInfo,
  numTrees, "auto", parameters._2, parameters._3, parameters._4)

     val rf = new RandomForestClassifier()
     .setLabelCol("indexedLabel")
     .setFeaturesCol("indexedFeatures")
     .setNumTrees(1)
     .setImpurity(parameters._2)
     .setMaxDepth(parameters._3)
     .setMaxBins(parameters._4)
     // Chain indexers and forest in a Pipeline
     val pipeline = new Pipeline()
     .setStages(Array(labelIndex, featureIndex, rf, labelConvert))
     val model = pipeline.fit(trainDataset)
     model
        }
// training all
  def training() : Unit  = {
    logger.info("..........Training...............")
    for( i <- 1 to numClassifiers.toInt){
      metaClassifier += trainingTree()
    }
  }

  /* private def reduceFunction(p1: RDD[(Double, SparseVector,DenseVector)],
      p2: RDD[(Double, SparseVector,DenseVector)]):
      RDD[(Double, SparseVector,DenseVector)] ={
    val rddZip=customZip(p1,p2)
    val rddOne=rddZip.map(ele=>{
      val probabilidadClase1Array=Array(ele._1._3(0),ele._2._3(0))
      val probabilidadClase2Array=Array(ele._1._3(1),ele._2._3(1))
      val probabilidadClase1=probabilidadClase1Array.sum/probabilidadClase1Array.size
      val probabilidadClase2=probabilidadClase2Array.sum/probabilidadClase2Array.size
      val probabilidadArray=new DenseVector(Array(probabilidadClase1,probabilidadClase2))
      // (label, features, probabilida(clase1, clase2))
      (ele._1._1,ele._1._2,probabilidadArray)
    })
    rddOne
  }*/

  private def reduceDF(p1: DataFrame, p2: DataFrame): DataFrame ={
    val nVar=Array("id","probability")
    val namesNew=nVar.map(name => col(name).as(name+"dos"))
    // join of the dataframes
    val dftoJoin=p2.select(namesNew : _*)
    var dfRetorno=p1.join(dftoJoin,p1("id")===dftoJoin("id"+"dos")).drop("iddos").
    rdd.map(row=>(row.getLong(0), row.getAs[DenseVector](3),row.getAs[DenseVector](4)))
    val rddInt=dfRetorno.map(row=>{
       val id=row._1
       val pro1=row._2
       val pro2=row._3
       val probabilidadClase1Array=Array(pro1(
         0),pro2(0))
       val probabilidadClase2Array=Array(pro1(1),pro2(1))
       val probabilidadClase1=probabilidadClase1Array.sum/probabilidadClase1Array.size
       val probabilidadClase2=probabilidadClase2Array.sum/probabilidadClase2Array.size
       val proVect=Array(probabilidadClase1,probabilidadClase2)//.toSparse
       var predLabel= if (probabilidadClase1>probabilidadClase2) -1.0 else 1.0
       val retorno=(id,proVect,predLabel)
       retorno
     })
     import sqlContext.implicits._
     val result=rddInt.toDF("id","probability","predictedLabel")
     result.write.mode(SaveMode.Overwrite).saveAsTable("reducedf")
     val dataframeResult=sqlContext.sql("SELECT * FROM reducedf")
    dataframeResult
  }


  def getPredictions(testData:DataFrame) : DataFrame ={
    logger.info("......... Preparing the dataframe of input...............")
    var test=testData.withColumn("id",monotonically_increasing_id())
    test.write.mode(SaveMode.Overwrite).saveAsTable("test")
    test = sqlContext.sql("SELECT * FROM "+ "test").coalesce(numPartitions).persist()
    val nConstant=Array("id","label","features")
    val namesConstant=nConstant.map(name => col(name))
    // coleccion de rdd con los resultados de las predicciones ((label, features, probabilida(clase1, clase2))
    logger.info("..........getting the results of all the trees...............")
    var coleccionPredRdd=metaClassifier.map(mod => mod.transform(test))
    //.rdd
    //.map(row=>(row.getDouble(0),row.getAs[SparseVector](1),row.getAs[DenseVector](2))))
    // RDD vacio donde se alamcenara los resultados de las predicciones
    // se ejecuta promedio sobbre las probabilidades
    //var predRddRes : RDD[(Double, SparseVector,DenseVector)] = sc.emptyRDD
    // se recorren los RDD con las predicciones de cada arbol
  //  coleccionPredRdd.reduce(_ join _)
  var predRddRes : DataFrame = sqlContext.emptyDataFrame
  //for (rddPred<-coleccionPredRdd) {
    // si es vacio se adiciona si no se llama reduceFunction
    //predRddRes=if(predRddRes.isEmpty) rddPred else reduceDF(predRddRes,rddPred)
    //}
    logger.info("..........getting the predictions to each tree...............")
    for (rddPred<-coleccionPredRdd) {
      // si es vacio se adiciona si no se llama reduceFunction
      predRddRes=if( predRddRes.count<=0) rddPred.select("id","probability","predictedLabel") else reduceDF(predRddRes,rddPred)
      }
     predRddRes.write.mode(SaveMode.Overwrite).saveAsTable("predictions")

     val dfREsultpredRddRes=sqlContext.sql("SELECT * FROM "+ "predictions").coalesce(numPartitions)
     dfREsultpredRddRes
  }

  def GetFeatureImportances():Vector={

    var allImportances=metaClassifier.map(mod => mod.featureImportances())

  }



/*
root
 |-- label: double (nullable = true)   SI
 |-- features: vector (nullable = true) SI
 |-- indexedLabel: double (nullable = true)
 |-- indexedFeatures: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)  SI
 |-- prediction: double (nullable = true)
 |-- predictedLabel: string (nullable = true) SI
*/


 }
