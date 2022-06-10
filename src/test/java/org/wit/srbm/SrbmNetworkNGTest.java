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

import java.io.IOException;

/**
 * @author koperix
 */
public class SrbmNetworkNGTest {

    public SrbmNetworkNGTest() {
    }

    @org.testng.annotations.BeforeClass
    public static void setUpClass() throws Exception {
    }

    @org.testng.annotations.AfterClass
    public static void tearDownClass() throws Exception {
    }

    @org.testng.annotations.BeforeMethod
    public void setUpMethod() throws Exception {
    }

    @org.testng.annotations.AfterMethod
    public void tearDownMethod() throws Exception {
    }

    /**
     * Test of main method, of class SRBM.
     */


    @Test
    public void testDBN() throws IOException, InterruptedException {
        SRBM v1 = new SRBMMapReduceJSA(
                new Configuration()
                        .setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(784)
                        .setNumberOfEpochs(10));
        SRBM v2 = new SRBMMapReduceJSA(v1,
                new Configuration()
                        .setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(784 )
                        .setNumberOfEpochs(10));
        SRBM v3 = new SRBMMapReduceJSA(v2,
                new Configuration()
                        .setBatchSize(200)
                        .setNumdims(784)
                        .setNumhid(784/28)
                        .setNumberOfEpochs(100));
        v1.train();
    }

}
