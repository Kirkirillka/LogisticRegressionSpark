import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.sql.DataFrame

import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer}
import org.apache.spark.ml.linalg.Vectors

  object utils {

    def disableLogs(): Unit ={
      // Suppress unessesary log output
      Logger.getLogger("org").setLevel(Level.OFF)
      Logger.getLogger("akka").setLevel(Level.OFF)
    }

    def extractFeatures(dataset:DataFrame): DataFrame ={
      val assembler = new VectorAssembler()
        .setInputCols(Array("pregnancy","glucose","arterial pressure","thickness of TC","insulin","body mass index","heredity","age"))
        .setOutputCol("features")

      val output = assembler.transform(dataset)
      //output.show(10)
      return output
    }

    def labelData(dataset:DataFrame): DataFrame ={

      val labeledTransformer = new StringIndexer()
        .setInputCol("diabet")
        .setOutputCol("label")

      val labeledFeatures = labeledTransformer.fit(dataset).transform(dataset)

      //labeledFeatures.show(10)

      return labeledFeatures
    }

    def rootMeanSquareError(dataset: DataFrame):Double = {
      val rdd = dataset.select("label","prediction").rdd

      val error = Math.sqrt(rdd.map(row=>(row(0).toString.toFloat,row(1).toString.toFloat))
        .map(r=>Math.pow(r._1-r._2, 2)).sum() / rdd.count())

      return error
    }

  }