/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.srbm;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.testng.annotations.Test;
import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.SrbmMapReduce;
import org.wit.snr.nn.srbm.visualization.*;

import java.io.IOException;

/**
 * @author koperix
 */
public class SrbmNetworkNGTest {


  /*  @Test
    public void testDBN2() throws IOException, InterruptedException, ClassNotFoundException {
        SrbmLayer v1 = new SrbmMapReduce(new RbmCfg().showViz(false));
        v1.load("/dane/2v2-srbm-layer.data");
        Matrix filter = v1.W().splitToColumnVectors().get(6).reshape(28);
        //MatrixRendererIF r = new WeightsInFrame(v1.W());
        //MatrixRendererIF r2 = new OneMatrixInFrame(filter);
        MatrixRendererIF r3 = new ManyMatrixInFrame(v1.W().splitToColumnVectors().stream().map(m -> m.reshape(28).transpose()).collect(Collectors.toList()));
        //r.render();
        //r2.render();
        r3.render();
        Thread.sleep(Long.MAX_VALUE);
    }

    @Test
    public void testDBN() throws IOException, InterruptedException, ClassNotFoundException {
        TrainingSetMinst tset = new TrainingSetMinst();
        List<List<Double>> x = tset.getSamples();

        SrbmLayer v1 = new SrbmMapReduce(
                new RbmCfg().setBatchSize(200)
                            .numdims(784)
                            .numhid(784)
                            .setSparsneseFactor(0.5)
                            .setNumberOfEpochs(15)
                            .setAcceptedError(0.03)
        );
        v1.load("/dane/2v1-srbm-layer.data");
        SrbmLayer v2 = new SrbmMapReduce(v1, new RbmCfg().setBatchSize(200)
                                                         .numdims(784)
                                                         .numhid(500)
                                                         .setNumberOfEpochs(10)
                                                         .setAcceptedError(0.0035)
        );
        v2.load("/dane/2v2-srbm-layer.data");
        SrbmLayer v3 = new SrbmMapReduce(v2, new RbmCfg().setBatchSize(200)
                                                         .numdims(500)
                                                         .numhid(10)
                                                         .setNumberOfEpochs(10)
                                                         .setAcceptedError(0.001)
        );
        v3.load("/dane/2v3-srbm-layer.data");


        v1.fit(x);
        v1.persist("/dane/2v1-srbm-layer.data");
        v2.persist("/dane/2v2-srbm-layer.data");
        v3.persist("/dane/2v3-srbm-layer.data");
    }

    @Test
    public void testRBM() throws IOException, InterruptedException, ClassNotFoundException {
        TrainingSetMinst tset = new TrainingSetMinst();
        List<List<Double>> x = tset.getSamples();
        SrbmLayer v1 = new SrbmMapReduce(
                new RbmCfg().setBatchSize(10)
                            .numdims(784)
                            .numhid(784 / 16)
                            .setSparsneseFactor(0.1)
                            .setNumberOfEpochs(1)
                            .setAcceptedError(0.04)
        );
        v1.fit(x);
        v1.persist("/dane/v1-single-srbm-layer.data");
    }

    @Test
    public void autoencoder() throws IOException, InterruptedException, ClassNotFoundException, CloneNotSupportedException, IllegalAccessException {

        TrainingSetMinst tset = new TrainingSetMinst();
        tset.load("C:\\Users\\artur.koperkiewicz\\Downloads\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
        List<List<Double>> x = tset.getSamples();
        x =x.subList(0,30000);

        int[] topology = {784, 200, 400, 20,
                          10,
                          20, 400, 200, 784};
        RbmCfg cfg = new RbmCfg()
                .setBatchSize(20)
                .learningRate(0.01)
                .setSparsneseFactor(0.65)
                .setNumberOfEpochs(30)
                .setAcceptedError(0.004)
                .persist(true)
                .setSaveVisualization(false)
                .showViz(false)
                .workDir("/dane/");
        DbnAutoencoder autoencoder = new DbnAutoencoder("autoencoder", cfg, topology);
        autoencoder.buildTopology();
        autoencoder.fit(x);
        Thread.sleep(Long.MAX_VALUE);
    }*/

    @Test
    void singleRBM() throws IOException, InterruptedException {


        var cfg = new RbmCfg()
                .setBatchSize(20)
                .learningRate(0.01)
                .setSparsneseFactor(0.65)
                .setNumberOfEpochs(30)
                .setAcceptedError(0.004)
                .persist(true)
                .setSaveVisualization(false)
                .showViz(false)
                .workDir("/dane/");
        var srbm = new SrbmMapReduce(cfg);
        RDD<Vector> trainigDataSet = getTrainigSetRDD();
        srbm.fit(trainigDataSet);




    }

    private RDD<Vector> getTrainigSetRDD() {
        SparkConf sparkConf = new SparkConf().setAppName("Read Text to RDD")
                                             .setMaster("local[8]")
                                             .set("spark.executor.memory", "4g");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        var trainingSet =
                sc.textFile("/home/artur/Pobrane/mnist/mnist_train.csv")
                            .map(csvLineString -> {
                                var imageData = csvLineString.split(",");
                                     double[] values = new double[imageData.length - 1];
                                     for (int i = 0; i < values.length; i++) {
                                        values[i] = Double.parseDouble(imageData[i + 1]);
                                    }
                                   return Vectors.dense(values);
                            });
        return trainingSet.rdd();
    }

}
