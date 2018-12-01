import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("SepalLength","SepalWidth","PetalLength","PetalWidth","class")
val newcol = when($"class".contains("Iris-setosa"), 1.0).
  otherwise(when($"class".contains("Iris-virginica"), 3.0).
  otherwise(2.0))
val newdf = df.withColumn("ID", newcol)

newdf.select("ID","SepalLength","SepalWidth","PetalLength","PetalWidth","class").show(150, false)
val assembler = new VectorAssembler()  .setInputCols(Array("SepalLength","SepalWidth","PetalLength","PetalWidth","ID")).setOutputCol("features")
val features = assembler.transform(newdf)
features.show(5)
val bkmeans = new BisectingKMeans().setK(2).setSeed(1)
val model = bkmeans.fit(features)
val cost = model.computeCost(features)
println(s"Within Set Sum of Squared Errors = $cost")
println("Cluster Centers: ")
val centers = model.clusterCenters
centers.foreach(println)
