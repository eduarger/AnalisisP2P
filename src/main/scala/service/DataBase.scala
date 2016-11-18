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
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, StringIndexerModel, VectorIndexerModel,VectorAssembler}
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector,Vectors}
import org.apache.spark.mllib.regression.LabeledPoint


class DataBase(tableBase: String,  numP:Int, sqlContext: SQLContext) extends Serializable {

  private val cols="""idn,resp_code,fraude,nolabel,monto,valida_cifin,1_max_monto_tarjetac,1_max_monto_email,
  1_promedio_monto_tarjetac,1_promedio_monto_email,coc_act_max_tarjeta,coc_act_pro_tarjeta,
  coc_act_max_email,coc_act_pro_email,COP ,OTH ,USD ,EUR  ,fds ,morn ,tard ,noch ,madr ,
  30_contar_dias_retail_code_tarjetac, 30_contar_dias_retail_code_email,probabilidad_franjaindex_tarjetac,probabilidad_franjaindex_email,
  probabilidad_ubicacion_tarjetac, probabilidad_ubicacion_email,probabilidad_id_comercio_tarjetac,probabilidad_id_comercio_email,
  probabilidad_id_sector_tarjetac, probabilidad_id_sector_email,probabilidad_ip_tarjetac,probabilidad_ip_email,
  probabilidad_retail_code_tarjetac,probabilidad_retail_code_email,probabilidad_nav_tarjetac,probabilidad_nav_email,probabilidad_os_tarjetac,
  probabilidad_os_email,1_promedio_cuotas_tarjetac,6_h_pro_tarjetac, 24_h_pro_tarjetac,15_d_pro_tarjetac,30_d_pro_tarjetac,60_d_pro_tarjetac,
  365_d_pro_tarjetac,6_h_can_tarjetac,24_h_can_tarjetac,15_d_can_tarjetac,  30_d_can_tarjetac,60_d_can_tarjetac,365_d_can_tarjetac,
  6_h_pro_email,24_h_pro_email,15_d_pro_email,30_d_pro_email,60_d_pro_email,365_d_pro_email,6_h_can_email,24_h_can_email,15_d_can_email,
  30_d_can_email,60_d_can_email,365_d_can_email,6_contar_horas_monto_ip,24_contar_horas_monto_ip,10_contar_min_monto_ip,
  coc_mon_cuo,coc_cuotas_cuotaspro,first_tarjeta,first_email,first_doc,emailnull,ubicacionull,tarjetanull,sectionnull,comercionull,
  punto_de_venta,documento_clientenull, confiabilidad_documento, nat_flag,1_max_monto_documentoclientec,1_promedio_monto_documentoclientec,coc_act_max_doc,
  coc_act_pro_doc,probabilidad_franjaindex_documentoclientec, probabilidad_ubicacion_documentoclientec,probabilidad_id_comercio_documentoclientec,
  probabilidad_id_sector_documentoclientec,probabilidad_ip_documentoclientec, probabilidad_retail_code_documentoclientec,
  cuenta_monto_cp40_bine_6horas,cuenta_monto_cp40_bine_24horas,cuenta_monto_cp40_bine_15horas,cuenta_monto_cp40_bine_30horas,
  cuenta_monto_cp40_bine_60horas,cuenta_monto_cp40_bine_365horas,probabilidad_nav_documentoclientec,probabilidad_os_documentoclientec,
  probabilidad_nat_flag_retail_code_6h,  probabilidad_nat_flag_retail_code_24h,probabilidad_nat_flag_retail_code_15d,
  probabilidad_nat_flag_retail_code_30d,probabilidad_nat_flag_retail_code_60d,  cuenta_monto_documentoclientec_6h,pro_monto_documentoclientec_6h,
  cuenta_monto_documentoclientec_24h,pro_monto_documentoclientec_24h,cuenta_monto_documentoclientec_15d, pro_monto_documentoclientec_15d,
  cuenta_monto_documentoclientec_30d,pro_monto_documentoclientec_30d,cuenta_monto_documentoclientec_60d,pro_monto_documentoclientec_60d,
  cuenta_monto_documentoclientec_365d,pro_monto_documentoclientec_365d,coc_6h_pro_nat_retail_code,coc_24h_pro_nat_retail_code,coc_15d_pro_nat_retail_code,coc_30d_pro_nat_retail_code,
  coc_60d_pro_nat_retail_code,coc_6h_pro_int_retail_code,coc_24h_pro_int_retail_code,coc_15d_pro_int_retail_code,
  coc_30d_pro_int_retail_code, cuenta_retail_code_documentoclientec_30d,coc_6h_can_tx_rec_apr_retail,coc_24h_can_tx_rec_apr_retail,
  coc_30d_can_tx_rec_apr_retail,coc_60d_can_tx_rec_apr_retail,coc_6h_sum_tx_rec_apr_retail, coc_24h_sum_tx_rec_apr_retail,
  coc_30d_sum_tx_rec_apr_retail,coc_60d_sum_tx_rec_apr_retail,coc_60d_pro_int_retail_code,max_match,match_tarjeta_documento_cliente,cant_same_data"""

  val query="SELECT " + cols+ " FROM "+ tableBase
  private val dataFrameBase=sqlContext.sql(query).coalesce(numP)
  @transient private val logger = LogManager.getLogger("DataBase")


  def getDataFrame():DataFrame={
    val retorno=dataFrameBase
    retorno
  }

  def getDataFrameNum(preCal:Boolean):DataFrame={
    val tName=tableBase+"_num"
    val retorno={
      if(preCal)
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)
      else
        dataFrameBase.drop("resp_code").drop("fraude").drop("nolabel").write.mode(SaveMode.Overwrite).saveAsTable(tName)
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)
    }
    retorno
  }

  def getDataFrameLabeledLegalFraud(preCal:Boolean):DataFrame={
    val tName=tableBase+"_labeled"
    val base=dataFrameBase.where("nolabel!=1")
    val names = base.columns
    val lon=names.length
    val ignore = Array("idn", "resp_code","fraude","nolabel")
    val assembler = (new VectorAssembler()
    .setInputCols( for (i <- names if !(ignore contains i )) yield i)
    .setOutputCol("features"))
    val retorno={
      if(preCal){
        logger.info("........Reading  "+tName +"..............")
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)

      } else{
        logger.info("........Converting to features...............")
        val data = assembler.transform(base)
        logger.info("..........Conviertinedo DF a labeling...............")
        val rows: RDD[Row] = data.rdd
        val labeledPoints: RDD[LabeledPoint]=(rows.map(row =>{LabeledPoint(row.getInt(2).toDouble,
        row.getAs[SparseVector](lon))}))
        import sqlContext.implicits._
        val labeledDF=labeledPoints.toDF()
        logger.info("........writing  "+tName +"..............")
        labeledDF.write.mode(SaveMode.Overwrite).saveAsTable(tName)
        logger.info("........Reading  "+tName +"..............")
        sqlContext.sql("SELECT * FROM "+tName).coalesce(numP)
   }
  }
  retorno



}

}
