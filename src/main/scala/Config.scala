package config
import java.io.File
case class Config(par: Int = 250, read: String = "", out: String = "sal",
  kfolds: Int = 5, trees: Seq[Int] = Seq(), imp: Seq[String] = Seq("entropy"),
  depth: Seq[Int] = Seq(10),bins: Seq[Int] = Seq(64))