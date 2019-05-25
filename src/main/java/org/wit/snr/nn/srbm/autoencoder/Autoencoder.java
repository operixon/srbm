package org.wit.snr.nn.srbm.autoencoder;

import org.wit.snr.nn.srbm.Configuration;
import org.wit.snr.nn.srbm.SRBM;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;
import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.io.IOException;

public class Autoencoder {

    private final Configuration cfgA;
    private final Configuration cfgB;
    private final SRBM netA;
    private final SRBM netB;


    public Autoencoder(Configuration cfga, Configuration cfgb) throws IOException, InterruptedException {
        this.cfgA = cfga;
        this.cfgB = cfgb;
        netA = new SRBMMapReduceJSA(cfgA);
        netB = new SRBMMapReduceJSA(cfgB);
        netB.connectPreviousLayer(netA);
    }

    public void go()
    {
        netA.train();
        Matrix W1 = netA.getLayer().W.transpose();
        netB.getLayer().W = W1;
        netB.train();

    }
}
