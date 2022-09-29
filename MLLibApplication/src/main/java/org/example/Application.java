package org.example;


import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Hello world!
 *
 */
public class Application
{
    public static void main( String[] args ) {

       SparkSession sparkSession = SparkSession.builder().appName("MLLibApplication").master("local").getOrCreate();
       Dataset<Row> rawData = sparkSession.read().format("csv").option("header",true).option("inferSchema",true).load("mllibData.csv");
       System.out.println(rawData);
       rawData.show();

        Dataset<Row>secondRawData = sparkSession.read().format("csv").option("header",true).option("inferSchema",true).load("eighteen.csv");

       //Spark MLLib Linear Regression Model
            System.out.println("Hello World !");
       VectorAssembler featuresVector = new VectorAssembler().setInputCols(new String[]{"Ay"}).setOutputCol("features");
        System.out.println("Hello World !");
       Dataset<Row> trasformedData = featuresVector.transform(rawData);
       Dataset<Row> secondData = featuresVector.transform(secondRawData);

       System.out.println(trasformedData);
       trasformedData.show();

        Dataset<Row> finalData = trasformedData.select("features","Satis");

        Dataset<Row>[]datasets = finalData.randomSplit(new double[]{0.75,0.25}); //train-test splitting
        Dataset<Row> trainData = datasets[0];
        Dataset<Row> testData = datasets[1];
        System.out.println("Train Data: ");
        trainData.show();

        System.out.println("Test Data: ");
        testData.show();


        LinearRegression linearRegression = new LinearRegression();
        linearRegression.setLabelCol("Satis"); //linear regression modelinin tahmin etmesi gerek kolon adı.
        LinearRegressionModel model = linearRegression.fit(trainData);

        /*Dataset<Row>transformTest = model.transform(secondData);
        Dataset<Row> testTrasform = model.transform(testData);

        testTrasform.show();
        transformTest.show();*/

        LinearRegressionTrainingSummary linearRegressionTrainingSummary = model.summary();

        System.out.println("*****START OF STATISTICS***************************");
        System.out.println("Statistics about the linear regression model: ");
        System.out.println("Root Mean Squared Error: " + linearRegressionTrainingSummary.rootMeanSquaredError());
        System.out.println("R-Squared Value: " + linearRegressionTrainingSummary.r2());
        System.out.println("Mean Absolute Error: "+linearRegressionTrainingSummary.meanAbsoluteError());
        System.out.println("Degrees of Freedom: "+linearRegressionTrainingSummary.degreesOfFreedom());
        System.out.println("*****END OF STATISTICS***************************");
















    }
}
