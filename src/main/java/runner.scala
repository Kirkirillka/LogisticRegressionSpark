import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql._
import utils.disableLogs
import utils.{extractFeatures, labelData,rootMeanSquareError}
import schema.dataSetSchema

object runner {
  def main(args: Array[String]): Unit = {

    disableLogs()

    var session = SparkSession
      .builder()
      .appName("spark-log-regression")
      .master("local[*]")
      .getOrCreate()

    var filePath = "data/diabets.csv"

    var df = session.read
      .option("header","false")
      .option("delimiter",",")
      .schema(dataSetSchema)
      .csv(filePath)

    df.printSchema()
    df.show(10)

    var features = extractFeatures(df)
    val labeled_features = labelData(features)

    var splits = labeled_features.randomSplit(Array(0.6,0.4), seed=12L)
    val trainingData = splits(0)
    val testData = splits(1)

    // trainingData.show(2)
    // testData.show(2)


    // Logistic Regression

    var lr = new LogisticRegression()
      .setMaxIter(1000000)
      .setRegParam(0.000001)
      .setElasticNetParam(0.007)

    // Train model
    val model = lr.fit(trainingData)

    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


    // Make prediction on the test data
    val prediction = model.transform(testData)
    //prediction.select("label","prediction").show(100)


    println(s"Manual Accuracy: ${rootMeanSquareError(prediction)}")

    // Evaluate the precision and recall
    val countProve = prediction.where("label == prediction").count()
    val count = prediction.count()

    println(s"Count of true predictions: $countProve ;Total Count: $count")

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(prediction)

    println(s"Calculated Accuracy: ${accuracy}")


  }
}
