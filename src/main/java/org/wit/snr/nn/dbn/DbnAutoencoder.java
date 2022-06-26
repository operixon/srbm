package org.wit.snr.nn.dbn;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.SRBM;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

public class DbnAutoencoder {

    static Logger log = Logger.getLogger(DbnAutoencoder.class.getName());

    private final String name;
    private final RbmCfg cfg;
    private final int[] topology;
    private final List<SRBMMapReduceJSA> layers = new LinkedList<SRBMMapReduceJSA>();

    public DbnAutoencoder(String name, RbmCfg cfg, int[] topology) throws IllegalAccessException {

        if (topology.length < 3 || topology.length % 2 == 0) {
            throw new IllegalAccessException("Ten autoenkoder musi mieć minimum 3 warstwy i mieć sirodek symetrii");
        }
        this.name = name;
        this.cfg = cfg;
        this.topology = topology;
    }


    public void buildTopology() throws IOException, InterruptedException, CloneNotSupportedException {
        // na poczatek uczymy tylko pierwszą połowę enkodera
        for (int i = 0; i <= (topology.length - 1) / 2; i++) {
            RbmCfg newLayerCfg = ((RbmCfg) cfg.clone()).numdims(topology[i])
                                                       .numhid(topology[i + 1])
                                                       .name(this.name + "-" + i);
            SRBMMapReduceJSA newLayer = layers.size() == 0
                                        ? new SRBMMapReduceJSA(newLayerCfg)
                                        : new SRBMMapReduceJSA(layers.get(layers.size() - 1), newLayerCfg);
            layers.add(newLayer);
        }
        log.info("Autoencoder has ben prepared with " + layers.size() + " rbm machines. Total topology size is " + topology.length);
    }


    public void fit() throws CloneNotSupportedException, IOException, InterruptedException {
        // na poczatek uczymy tylko pierwszą połowę enkodera
        layers.get(0).train();
        log.info("Training of encoder end.");
        for (int i = (topology.length - 1) / 2; i < topology.length; i++) {
            RbmCfg newLayerCfg = ((RbmCfg) cfg.clone()).numdims(topology[i])
                                                       .numhid(topology[i + 1])
                                                       .name(this.name + "-" + i);
            SRBMMapReduceJSA newLayer = layers.size() == 0
                                        ? new SRBMMapReduceJSA(newLayerCfg)
                                        : new SRBMMapReduceJSA(layers.get(layers.size() - 1), newLayerCfg);
            layers.add(newLayer);
        }
        // podmieniamy flaki w warstwach mirror
        // warstwy w topologii sa nieparzyste, ale ilość rbm jest parysta
        for (int i = 0; i <= layers.size() / 2; i++) {
            SRBMMapReduceJSA baseLayer = layers.get(i);
            SRBMMapReduceJSA mirroredLayer = layers.get(layers.size() - i - 1);
            mirroredLayer.getLayer().W = baseLayer.getLayer().W.transpose();
            mirroredLayer.getLayer().vbias = baseLayer.getLayer().hbias;
            mirroredLayer.getLayer().hbias = baseLayer.getLayer().vbias;
        }
        log.info("Autoencoder has ben prepared with " + layers.size() + " rbm machines. Total topology size is " + topology.length);

    }
}
