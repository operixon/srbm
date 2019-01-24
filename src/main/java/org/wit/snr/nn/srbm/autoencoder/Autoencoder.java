package org.wit.snr.nn.srbm.autoencoder;

import org.wit.snr.nn.srbm.Configuration;
import org.wit.snr.nn.srbm.SRBM;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;

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


    }

    public void go()
    {
        netA.train();
        netB.train();
    }
}
