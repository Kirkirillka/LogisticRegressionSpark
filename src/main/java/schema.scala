import org.apache.spark.sql.types._

object schema {
  var dataSetSchema = StructType(Array(
    StructField("pregnancy", IntegerType, true),
    StructField("glucose", IntegerType, true),
    StructField("arterial pressure", IntegerType, true),
    StructField("thickness of TC", IntegerType, true),
    StructField("insulin", IntegerType, true),
    StructField("body mass index", DoubleType, true),
    StructField("heredity", DoubleType, true),
    StructField("age", IntegerType, true),
    StructField("diabet", IntegerType, true)))
}
