package ca.redsofa.jobs;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;

//Comments in this code are from Scala Data Analysis Cookbook (chapter 5 section on K-means)
public class KMeansApp 
{
    public static void main( String[] args )
    {
		SparkSession spark = SparkSession
				.builder()
				.appName("IRIS KMeans App")
				.config("spark.eventLog.enabled", "false")
				.config("spark.driver.memory", "2g")
				.config("spark.executor.memory", "2g")
				.enableHiveSupport()
				.getOrCreate();

		System.out.println("Starting job...");
        if (args.length == 0) {
            System.out.println("Must specify location of iris data file.");
            System.exit(-1);
        }
        
		String rawData = args[0];
		Dataset<String> sourceDS = spark.read().textFile(rawData);
		System.out.println("Source Data");
		sourceDS.show(10, false);

		Function<String, Vector> getVector = (String s) -> {			
			String[] elements = s.split(",");
			return Vectors.dense( new Double(elements[0]),new Double(elements[1]),new Double(elements[2]),new Double(elements[3]));
		};

		JavaRDD<Vector> nonScaledData = sourceDS.toJavaRDD().map(getVector);
		

		//Summary statistics before scaling
		MultivariateStatisticalSummary stats = Statistics.colStats(nonScaledData.rdd());		  
		System.out.println("Summary Statistics (No Scaling) : ");		
		System.out.format(" Max : %s\n Min : %s\n Mean : %s\n Variance : %s\n", stats.max(), stats.min(), stats.mean(), stats.variance());

		
		//Scale the data ..
		//It is always advisable to perform feature scaling before running a K-means:
		StandardScalerModel scaler = new org.apache.spark.mllib.feature.StandardScaler(true, true).fit(nonScaledData.rdd());
		//Since K-means goes through the dataset multiple times, caching the data is strongly recommended to avoid recomputation:
		JavaRDD<Vector> scaledData = scaler.transform(nonScaledData).cache();
		
		//Calculate summary Statistics on scaled data
		stats = Statistics.colStats(scaledData.rdd());
		System.out.println("Statistics after scaling :");		
		System.out.format(" Max : %s\n Min : %s\n Mean : %s\n Variance : %s\n", stats.max(), stats.min(), stats.mean(), stats.variance());

		//If the data is large, running the entire set of data just to obtain the number 
		//of clusters is computationally expensive.
		//Instead, we can take a random sample and come up with the k value.
		JavaRDD<Vector> scaledDataSample = scaledData.sample(false, 0.2).cache();

		KMeansModel model;
		int numberOfIterations = 5;
		
		int[] k = {1,2,3,4,5,6,7,8,9};
		
		for (int i = 0; i<= k.length - 1; i++ ){
			System.out.println("Number of Clusters : " + k[i]);
			model = KMeans.train(scaledDataSample.rdd(), k[i], numberOfIterations, -1, KMeans.K_MEANS_PARALLEL());
			System.out.println("Within Set Sum Error " + model.computeCost(scaledDataSample.rdd()));			
		}
		
		//Number of clusters is 3. It's the point at wich the cost does not reduce significantly. Point is called 
		//an elbow bend.
		//Now that we have figured out the number of clusters, let's run the algorithm against the entire dataset
		model = KMeans.train(scaledData.rdd(), 3, numberOfIterations);
		
		System.out.println("Predicting a point. Input Data :[4.9,3.5,1.4,0.2] : ");


		Vector  points = Vectors.dense(4.9,3.5,1.4,0.2);
		Vector scaledPoints = scaler.transform(points);
		
		System.out.println("Predicted Cluster : " + model.predict(scaledPoints));
		
		System.out.println("Predicting a bunch of points... :");
		JavaRDD<Integer> predictions = model.predict(scaledData);
		
		for (Integer i : predictions.collect()){
			System.out.println("Predicted Cluster : " + i);
		}
		
		//The cost is the square of the distance between all points in a cluster to its centroid. 
		//Therefore, a good model must have the least cost.
		System.out.println("Total cost " + model.computeCost(scaledDataSample.rdd()));
		
		System.out.println("Centers : ");
		Vector[] centers = model.clusterCenters();  
		
		for (Vector v : centers){
			System.out.println("Center : " + v);			
		}
		//Save the model (for future use)
		model.save(spark.sparkContext(),"MyModel");
		//Load the saved model (as a test)
		KMeansModel savedModel = KMeansModel.load(spark.sparkContext(), "MyModel");
		//make a prediction with saved model
		System.out.println("Saved Model Predicted Cluster : " + savedModel.predict(scaledPoints));
    }
}