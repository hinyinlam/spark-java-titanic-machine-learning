package titanic.io.demo;

import lombok.Data;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.count;

@SpringBootApplication
public class DemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
        //We don't necessary run the Machine learning along side Spring,
        //it's just the ecosystem of Spring allows us to further develop other application functions
    }

    @Bean
    public static CommandLineRunner runme() {
        return args -> {
            DemoApplication app = new DemoApplication();
            app.runSparkML();
        };
    }

    public void runSparkML(){
        SparkSession ss = SparkSession.builder().appName("TitanicSurvive").config("spark.eventLog.enabled", "false").master("local[1]").getOrCreate();
        ss.sparkContext().setLogLevel("ERROR");
        Dataset<Row> rows = ss.read().option("header", true).csv("train.csv");

        //Selection of useful columns (features)
        System.out.println("==========Select useful colmns and casting========");
        rows = rows.select(
                col("Survived").cast(DataTypes.IntegerType),
                col("Pclass").cast(DataTypes.DoubleType),
                col("Sex"),
                col("Age").cast(DataTypes.DoubleType),
                col("Fare").cast(DataTypes.DoubleType),
                col("Embarked")
        );

        //Rows now has the right type to be typed
        System.out.println("Scheme:");
        rows.printSchema();

        //How the deta looks like
        System.out.println("==========Preliminary Statistic==========");
        System.out.println("Total passenger count: " + rows.count());
        Dataset<Row> describe = rows.describe();
        describe.show();

        //Data cleansing -- Study if Null is a problem (Yes, in our case)
        System.out.println("==========How many rows are with null value?==========");
        String colNames[] = rows.columns();
        Dataset<Row> summary = null;
        for(int i = 0; i < colNames.length; i++){
            String thisColName = colNames[i];
            Dataset<Row> numNullinCol = rows.filter(col(thisColName).isNull()).select(count("*").as("NullOf"+thisColName));
            /*
            To reader: Please suggest better way to do counting of null values in each columns using Spark Java API
             */
            if(summary==null){
                summary=numNullinCol;
                continue;
            }
            summary = summary.join(numNullinCol);
        }
        summary.show();

        //Data cleansing - Remove Null Values
        System.out.println("==========Eliminate Row with null values==========");
        Column dropCondition = null;
        for(int i = 0; i < colNames.length; i++){
            Column filterCol = col(colNames[i]).isNotNull();
           if(dropCondition==null) {
               dropCondition = filterCol;
               continue;
           }
           dropCondition = dropCondition.and(filterCol);
        }
        System.out.println("Filter condition: " + dropCondition);
        Dataset<Row> nonNullPassengers = rows.filter(dropCondition);
        System.out.println("Remain number of non-null passenger: " + nonNullPassengers.count());
        nonNullPassengers.show();

        //Feature Engineering -- Prepare
        System.out.println("========Feature preparation - Index String to numbers for Spark MLib========");
        StringIndexer sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("Gender").setHandleInvalid("keep");
        StringIndexer embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("Boarded").setHandleInvalid("keep");

        Dataset<Row> genderPassengers = sexIndexer.fit(nonNullPassengers).transform(nonNullPassengers);
        Dataset<Row> boardedGenderPassengers = embarkedIndexer.fit(genderPassengers).transform(genderPassengers);

        boardedGenderPassengers = boardedGenderPassengers.drop("Sex", "Embarked");

        boardedGenderPassengers.printSchema();
        boardedGenderPassengers.show();
        System.out.println("The column Gender and Boarded have been indexed to map to numbers");


        //Feature Engineering -- Prepare
        System.out.println("========Feature engineering ========");
        String featureCol[] = {
                "Pclass",
                "Age",
                "Fare",
                "Gender",
                "Boarded"
        };
            //Be aware of the mistake to pass the label(ie observed result) "Survived" column, model will be 100% correct if so...too good to be true

        VectorAssembler vAssembler = new VectorAssembler();
        vAssembler.setInputCols(featureCol).setOutputCol("features");

        Dataset<Row> featureReadyDF = vAssembler.transform(boardedGenderPassengers);

        featureReadyDF.printSchema();
        featureReadyDF.show();


        //Modeling -- Training data and Testing data
        System.out.println("========Modeling - Use same data set as both training and testing  -- NOT best practice ========");
        //Actually, we should use the "seen" data "train.csv" to train model and "unseen" "test.csv" for validation.
        //As we don't want to download the "test.csv" for the test part, we just split existing "train.csv" -> "80% train, 20% test" (pretend we haven't see 20% test data)
        //Because of simpliticy of this example, bias/look-ahead didn't happen;

        Dataset<Row>[] bothTrainTestDFs = featureReadyDF.randomSplit(new double[]{0.8d,0.2d});
        Dataset<Row> trainDF = bothTrainTestDFs[0];
        Dataset<Row> testDF = bothTrainTestDFs[1];

        System.out.println("===Training set===");
        trainDF.printSchema();
        trainDF.show();
        System.out.println("Total record: " + trainDF.count());

        System.out.println("===Testing set===");
        testDF.printSchema();
        testDF.show();
        System.out.println("Total record: " + testDF.count());

        //Modeling -- Machine learning (train an estimator)
        System.out.println("========Modeling - Building a model with an estimator ========");
        RandomForestClassifier estimator = new RandomForestClassifier()
                .setLabelCol("Survived")
                .setFeaturesCol("features")
                .setMaxDepth(5);

        RandomForestClassificationModel model = estimator.fit(trainDF);
            //we ask the Estimator to learn from training data
            //Now we have the learnt `model` to be used with test data and see how good this model is

        //Modeling -- Prediction with existing data (Given some features, what is the prediction?)
        System.out.println("========Modeling - Use an estimator to predict if the passenger survived==========");
        Dataset<Row> predictions = model.transform(testDF);

        System.out.println("Here is the predictions:");
        predictions.printSchema();
        predictions.show();

        //Modeling -- How good is our predictions?
        System.out.println("========Modeling - Check how good is the model we've built ==========");

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("Survived")
                .setPredictionCol("prediction");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy: " + accuracy);
    }


    //You might be tempted to use Dataset<Passenger> but since we are going to add/delete/move/transfer column around, this is not used
    @Data
    public static class Passenger{
        //Integer PassengerId;
        Integer Survived;
        Double Pclass;
        //String Name;
        String Sex;
        Double Age;
        //Integer SibSp;
        //Integer Parch;
        //Integer Ticket;
        Double Fare;
        //String Cabin;
        String Embarked;
    }

}
