package config
import java.io.File
case class Config(in:String="base_test",par: Int = 250, read: String = "", out: String = "sal",
  estrategia: String = "kmeans",mex: String = "8g", hmem: String = "600", kfolds: Int = 5,
  trees: Seq[Int] = Seq(), imp: Seq[String] = Seq("entropy"),
  depth: Seq[Int] = Seq(10),bins: Seq[Int] = Seq(64), axes: Seq[Int]=Seq(-10,10), filter: String ="")
