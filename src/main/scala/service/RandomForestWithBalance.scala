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
import scala.reflect.ClassTag
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
    labelMinor: Double
    sqlContext: SQLContext,
    sc: SparkContext
  )
    extends Serializable {

   val numPartitions=numP
   private var metaClassifier = scala.collection.mutable.ArrayBuffer.empty[PipelineModel]
   // training dataset empty
   private var trainDataset=sqlContext.createDataFrame(sc.emptyRDD[Row], baseTrain.schema)
   // relation for balance the training datasets
   val numOfMinorClass= this.baseTrain.where("label="+labelMinor).count()
   val numOfMayorClass= this.baseTrain.where("label!="+labelMinor).count()
   // calculate the smaple fraction and the number of classifiers
   private var sampleFraction=numOfMinorClass.toDouble/numOfMayorClass.toDouble
   private var numClassifiers=numOfMayorClass/numOfMinorClass
   // indexers and converters
   private val labelIndex=laIndx
   private val featureIndex=feIndx
   private val labelConvert=laConv
   @transient private val logger = LogManager.getLogger("RandomForestWithBalance")
   logger.info("Number of classifiers(trees): "+numClassifiers)
   logger.info("sample rate is: "+sampleFraction)

// set the training dataset
   def getTrainingBalancedSet() : Unit ={
     val dfMinor= this.baseTrain.where("label="+labelMinor)
     val dfMayor= this.baseTrain.where("label!="+labelMinor).sample(false,this.sampleFraction)
     this.trainDataset=dfMinor.unionAll(dfMayor)
   }
// traing one tree
   def trainingTree() : PipelineModel  = {
     // set the training dataset in each traiing
     getTrainingBalancedSet()
     val rf = new RandomForestClassifier()
     .setLabelCol("indexedLabel")
     .setFeaturesCol("indexedFeatures")
     .setNumTrees(1)
     .setImpurity(parameters._2)
     .setMaxDepth(parameters._3)
     .setMaxBins(parameters._4)
     // Chain indexers and forest in a Pipeline
     val pipeline = new Pipeline()
     .setStages(Array(this.labelIndex, this.featureIndex, rf, this.labelConvert))
     val model = pipeline.fit(this.trainDataset)
     model
        }
// training all
  def training() : Unit  = {
    logger.info("..........Training...............")
    for( i <- 1 to this.numClassifiers){
      metaClassifier += trainingTree()
    }
  }

  //

  def prepare[T: ClassTag](rdd: RDD[T], n: Int) =
  rdd.zipWithIndex.sortBy(_._2, true, n).keys

  def customZip[T: ClassTag, U: ClassTag](rdd1: RDD[T], rdd2: RDD[U]) = {
    val n = rdd1.partitions.size + rdd2.partitions.size
    prepare(rdd1, n).zip(prepare(rdd2, n))
  }

  private def reduceFunction(p1: RDD[(Double, SparseVector,DenseVector)],
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
  }




  @transient val calPredictedLabelDF  = (row:Row)=> {

     retorno
    }


  private def reduceDF(p1: DataFrame, p2: DataFrame): DataFrame ={
     val news=Array("id","probability")
     val namesNew=news.map(name => col(name).as(name+"dos"))
     val dftoJoin=p2.select(namesNew : _*)
     var dfRetorno=p1.join(dftoJoin,p1("id")===dftoJoin("id"+"dos")).drop("iddos").rdd.map(
       row=>(row.getLong(0), row.getDouble(1),row.getAs[SparseVector](2),
       row.getAs[DenseVector](3),row.getAs[DenseVector](4)))
     val rddInt=dfRetorno.map(row=>{
       val id=row._1
       val label=row._2
       val features=row._3
       val pro1=row._4
       val pro2=row._5
       val probabilidadClase1Array=Array(pro1(0),pro2(0))
       val probabilidadClase2Array=Array(pro1(1),pro2(1))
       val probabilidadClase1=probabilidadClase1Array.sum/probabilidadClase1Array.size
       val probabilidadClase2=probabilidadClase2Array.sum/probabilidadClase2Array.size
       val proVect=Array(probabilidadClase1,probabilidadClase2)//.toSparse
       var predLabel= if (probabilidadClase1>probabilidadClase2) -1.0 else 1.0
       val retorno=(id,features,label,predLabel,proVect)
       retorno
     })
     import sqlContext.implicits._
     val dataframeResult=rddInt.toDF("id","features", "label", "predictedLabel","probability")
     dataframeResult
  }


  val calPredictedLabel = (probability: DenseVector,class1:Int, class2:Int) => {
    var result={
      if (probability(0)>probability(1))
        class1
      else
        class2
    }
    result
               }


  def getPredictions(testData:DataFrame) : DataFrame ={
    var test=testData.withColumn("id",monotonically_increasing_id())
    test.write.mode(SaveMode.Overwrite).saveAsTable("test")
    test = sqlContext.sql("SELECT * FROM "+ "test").coalesce(numPartitions)
    val n=Array("id","label","features","probability")
    val names=n.map(name => col(name))
    // coleccion de rdd con los resultados de las predicciones ((label, features, probabilida(clase1, clase2))
    logger.info("..........checking all the trees...............")
    var coleccionPredRdd=this.metaClassifier.map(mod => mod.transform(test).select(names : _*))
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
    //predRddRes=if(predRddRes.isEmpty) rddPred else this.reduceDF(predRddRes,rddPred)
    //}

    for (rddPred<-coleccionPredRdd) {
      // si es vacio se adiciona si no se llama reduceFunction
      predRddRes=if( predRddRes.count<=0) rddPred else reduceDF(predRddRes,rddPred)
      }

     predRddRes



   //logger.info("..........getting predictedLabel...............")
   //val resultRDD: RDD[(Double, SparseVector,SparseVector,Int)]=(predRddRes

   //.map(e=>(e._1,e._2,e._3,calPredictedLabel(e._3,-1,1))))
   //logger.info("..........Creating dataframe...............")
   //creacion del dataframe y anadir la columna predictedLabel
   //import sqlContext.implicits._
   //val dataframeResult=resultRDD.toDF("label", "features", "probability","predictedLabel")
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
