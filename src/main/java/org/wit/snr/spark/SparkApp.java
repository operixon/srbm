package org.wit.snr.spark;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class SparkApp {




    public static void main(String[] args) {
        //
        // The "modern" way to initialize Spark is to create a SparkSession
        // although they really come from the world of Spark SQL, and Dataset
        // and DataFrame.
        //
        SparkSession spark = SparkSession
                .builder()
                .appName("RDD-Basic")
                .master("local[8]")
                .getOrCreate();

        //
        // Operating on a raw RDD actually requires access to the more low
        // level SparkContext -- get the special Java version for convenience
        //
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

        // put some data in an RDD
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        //
        // Since this SparkContext is actually a JavaSparkContext, the methods
        // return a JavaRDD, which is more convenient as well.
        //
        JavaRDD<Integer> numbersRDD = sc.parallelize(numbers, 4);
        System.out.println("*** Print each element of the original RDD");
        System.out.println("*** (they won't necessarily be in any order)");
        // Since printing is delegated to the RDD it happens in parallel.
        // For versions of Java without lambda, Spark provides some utility
        // interfaces like VoidFunction.
        numbersRDD.foreach(i -> System.out.println(i));

        // NOTE: in the above it may be tempting to replace the above
        // lambda expression with a method reference -- System.out::println --
        // but alas this results in
        //     java.io.NotSerializableException: java.io.PrintStream

        // Transform the RDD element by element -- this time use Function
        // instead of a Lambda. Notice how the RDD changes from
        // JavaRDD<Integer> to JavaRDD<double>.
        JavaRDD<Double> transformedRDD =
                numbersRDD.map(n -> new Double(n) / 10);

        // let's see the elements
        System.out.println("*** Print each element of the transformed RDD");
        System.out.println("*** (they may not even be in the same order)");
        transformedRDD.foreach(i -> System.out.println(i));

        // get the data back out as a list -- collect() gathers up all the
        // partitions of an RDD and constructs a regular List
        List<Double> transformedAsList = transformedRDD.collect();
        // interesting how the list comes out sorted but the RDD didn't
        System.out.println("*** Now print each element of the transformed list");
        System.out.println("*** (the list is in the same order as the original list)");
        for (Double d : transformedAsList) {
            System.out.println(d);
        }

        // explore RDD partitioning properties -- glom() keeps the RDD as
        // an RDD but the elements are now lists of the original values --
        // the resulting RDD has an element for each partition of
        // the original RDD
        JavaRDD<List<Double>> partitionsRDD = transformedRDD.glom();
        System.out.println("*** We _should_ have 4 partitions");
        System.out.println("*** (They can't be of equal size)");
        System.out.println("*** # partitions = " + partitionsRDD.count());
        // specifying the type of l is not required here but sometimes it's useful for clarity
        partitionsRDD.foreach((List<Double> l) -> {
            // A string for each partition so the output isn't garbled
            // -- remember the RDD is still distributed so this function
            // is called in parallel
            StringBuffer sb = new StringBuffer();
            for (Double d : l) {
                sb.append(d);
                sb.append(" ");
            }
            System.out.println(sb);
        });

        spark.stop();
    }


}
