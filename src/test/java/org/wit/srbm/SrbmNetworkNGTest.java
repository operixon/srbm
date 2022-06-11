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

import java.io.IOException;

/**
 * @author koperix
 */
public class SrbmNetworkNGTest {


    @Test
    public void testDBN() throws IOException, InterruptedException, ClassNotFoundException {

        //TrainigSet minst = Trainin

        SRBM v1 = new SRBMMapReduceJSA(
                new Configuration().setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(784)
                        .setSparsneseFactor(0.1)
                        .setNumberOfEpochs(10)
                        .setAcceptedError(0.04)
        );
        v1.load("/dane/v1-srbm-layer.data");
        SRBM v2 = new SRBMMapReduceJSA(v1,
                new Configuration()
                        .setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(784)
                        .setNumberOfEpochs(10)
                        .setAcceptedError(0.0035)
        );
        v2.load("/dane/v2-srbm-layer.data");
        SRBM v3 = new SRBMMapReduceJSA(v2,
                new Configuration()
                        .setBatchSize(10)
                        .setNumdims(784)
                        .setNumhid(784 / 28)
                        .setNumberOfEpochs(1)
                        .setAcceptedError(0.001)
        );
        v3.load("/dane/v3-srbm-layer.data");
        v1.train();
        v1.persist("/dane/v1-srbm-layer.data");
        v2.persist("/dane/v2-srbm-layer.data");
        v3.persist("/dane/v3-srbm-layer.data");
    }

}
