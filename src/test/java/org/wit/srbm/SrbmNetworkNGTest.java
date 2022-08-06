/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.srbm;

import org.testng.annotations.Test;
import org.wit.snr.nn.dbn.DbnAutoencoder;
import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.SRBM;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;
import org.wit.snr.nn.srbm.trainingset.TrainingSetMinst;
import org.wit.snr.nn.srbm.visualization.*;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author koperix
 */
public class SrbmNetworkNGTest {


    @Test
    public void testDBN2() throws IOException, InterruptedException, ClassNotFoundException {
        SRBM v1 = new SRBMMapReduceJSA(new RbmCfg().showViz(false));
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

        SRBM v1 = new SRBMMapReduceJSA(
                new RbmCfg().setBatchSize(200)
                            .numdims(784)
                            .numhid(784)
                            .setSparsneseFactor(0.5)
                            .setNumberOfEpochs(15)
                            .setAcceptedError(0.03)
        );
        v1.load("/dane/2v1-srbm-layer.data");
        SRBM v2 = new SRBMMapReduceJSA(v1, new RbmCfg().setBatchSize(200)
                                                       .numdims(784)
                                                       .numhid(500)
                                                       .setNumberOfEpochs(10)
                                                       .setAcceptedError(0.0035)
        );
        v2.load("/dane/2v2-srbm-layer.data");
        SRBM v3 = new SRBMMapReduceJSA(v2, new RbmCfg().setBatchSize(200)
                                                       .numdims(500)
                                                       .numhid(10)
                                                       .setNumberOfEpochs(10)
                                                       .setAcceptedError(0.001)
        );
        v3.load("/dane/2v3-srbm-layer.data");


        v1.train(x);
        v1.persist("/dane/2v1-srbm-layer.data");
        v2.persist("/dane/2v2-srbm-layer.data");
        v3.persist("/dane/2v3-srbm-layer.data");
    }

    @Test
    public void testRBM() throws IOException, InterruptedException, ClassNotFoundException {
        TrainingSetMinst tset = new TrainingSetMinst();
        List<List<Double>> x = tset.getSamples();
        SRBM v1 = new SRBMMapReduceJSA(
                new RbmCfg().setBatchSize(10)
                            .numdims(784)
                            .numhid(784 / 16)
                            .setSparsneseFactor(0.1)
                            .setNumberOfEpochs(1)
                            .setAcceptedError(0.04)
        );
        v1.train(x);
        v1.persist("/dane/v1-single-srbm-layer.data");
    }

    @Test
    public void autoencoder() throws IOException, InterruptedException, ClassNotFoundException, CloneNotSupportedException, IllegalAccessException {

        TrainingSetMinst tset = new TrainingSetMinst();
        //tset.load("C:\\Users\\artur.koperkiewicz\\Downloads\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
        tset.loadTestFromResources();
        List<List<Double>> x = tset.getSamples();
        x =x.subList(0,2000);

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
    }

}
