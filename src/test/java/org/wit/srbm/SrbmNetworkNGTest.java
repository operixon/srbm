/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.srbm;

import org.testng.annotations.Test;
import org.wit.snr.nn.srbm.Configuration;
import org.wit.snr.nn.srbm.SRBM;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;
import org.wit.snr.nn.srbm.TrainingSet;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.visualization.*;

import java.io.IOException;
import java.util.stream.Collectors;

/**
 * @author koperix
 */
public class SrbmNetworkNGTest {


    @Test
    public void testDBN2() throws IOException, InterruptedException, ClassNotFoundException {
        SRBM v1 = new SRBMMapReduceJSA(new Configuration().setShowVisualizationWindow(false));
        v1.load("/dane/2v2-srbm-layer.data");
        Matrix filter = v1.W().splitToColumnVectors().get(6).reshape(28);
        //MatrixRendererIF r = new WeightsInFrame(v1.W());
        //MatrixRendererIF r2 = new OneMatrixInFrame(filter);
        MatrixRendererIF r3 = new ManyMatrixInFrame(v1.W().splitToColumnVectors().stream().map(m->m.reshape(28).transpose()).collect(Collectors.toList()));
        //r.render();
        //r2.render();
        r3.render();
        Thread.sleep(Long.MAX_VALUE);
    }

    @Test
    public void testDBN() throws IOException, InterruptedException, ClassNotFoundException {


        SRBM v1 = new SRBMMapReduceJSA(
                new Configuration().setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(784)
                        .setSparsneseFactor(0.5)
                        .setNumberOfEpochs(15)
                        .setAcceptedError(0.03)
        );
        v1.load("/dane/2v1-srbm-layer.data");
        SRBM v2 = new SRBMMapReduceJSA(v1,
                new Configuration()
                        .setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(500)
                        .setNumberOfEpochs(10)
                        .setAcceptedError(0.0035)
        );
        v2.load("/dane/2v2-srbm-layer.data");
        SRBM v3 = new SRBMMapReduceJSA(v2,
                new Configuration()
                        .setBatchSize(200)
                        .setNumdims(500)
                        .setNumhid(10)
                        .setNumberOfEpochs(10)
                        .setAcceptedError(0.001)
        );
        v3.load("/dane/2v3-srbm-layer.data");


        v1.train();
        v1.persist("/dane/2v1-srbm-layer.data");
        v2.persist("/dane/2v2-srbm-layer.data");
        v3.persist("/dane/2v3-srbm-layer.data");
    }

    @Test
    public void testRBM() throws IOException, InterruptedException, ClassNotFoundException {


        SRBM v1 = new SRBMMapReduceJSA(
                new Configuration().setBatchSize(10)
                        .setNumdims(784)
                        .setNumhid(784/16)
                        .setSparsneseFactor(0.1)
                        .setNumberOfEpochs(1)
                        .setAcceptedError(0.04)
        );
        v1.train();
        v1.persist("/dane/v1-single-srbm-layer.data");
    }

    @Test
    public void autoencoder() throws IOException, InterruptedException, ClassNotFoundException {

        SRBM v1 = new SRBMMapReduceJSA(new Configuration().setShowVisualizationWindow(false));
        v1.load("/dane/2v2-srbm-layer.data");
        SRBM v2 = new SRBMMapReduceJSA(new Configuration().setShowVisualizationWindow(false));
        v2.load("/dane/2v2-srbm-layer.data");
        SRBM v3 = new SRBMMapReduceJSA(new Configuration().setShowVisualizationWindow(false));
        v3.load("/dane/2v2-srbm-layer.data");

        SRBM v3t = v3.autoencoderMirror();
        SRBM v2t = new SRBMMapReduceJSA(new Configuration().setShowVisualizationWindow(false));
        SRBM v1t = new SRBMMapReduceJSA(new Configuration().setShowVisualizationWindow(false));

        v3t.


        v1.setNext(v2).setNext(v3).setNext(v3t).setNext(v2t).setNext(v1t);


        Matrix filter = v1.W().splitToColumnVectors().get(6).reshape(28);
        //MatrixRendererIF r = new WeightsInFrame(v1.W());
        //MatrixRendererIF r2 = new OneMatrixInFrame(filter);
        MatrixRendererIF r3 = new ManyMatrixInFrame(v1.W().splitToColumnVectors().stream().map(m->m.reshape(28).transpose()).collect(Collectors.toList()));
        //r.render();
        //r2.render();
        r3.render();
        Thread.sleep(Long.MAX_VALUE);
    }

}
