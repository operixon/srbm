package org.wit.snr.nn.dbn;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.SRBM;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;
import org.wit.snr.nn.srbm.math.collection.Matrix;

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


    public void buildTopology() throws IOException, InterruptedException, CloneNotSupportedException, ClassNotFoundException {
        for (int i = 0; i < topology.length - 1; i++) {
            RbmCfg newLayerCfg = ((RbmCfg) cfg.clone()).numdims(topology[i])
                                                       .numhid(topology[i + 1])
                                                       .name(this.name + "-" + i);
            SRBMMapReduceJSA newLayer = layers.size() == 0
                                        ? new SRBMMapReduceJSA(newLayerCfg)
                                        : new SRBMMapReduceJSA(layers.get(layers.size() - 1), newLayerCfg);
            layers.add(newLayer);
        }
        if (cfg.load()) {
            for (int i = 0; i < layers.size(); i++) {
                layers.get(i).load(cfg.workDir() + "/" + name + "-" + i);
            }
        }
    }



    public void fit()  {
        // na poczatek uczymy tylko pierwszą połowę enkodera
        for (int i=0; i < layers.size() / 2; i++) {
            layers.get(i).train();
        }
        log.info("Training of encoder end.");
        // podmieniamy flaki w warstwach mirror
        // warstwy w topologii sa nieparzyste, ale ilość rbm jest parysta
        copyModelFromEncoderToDecoder();
        persistAutoencoderModel();
        log.info("Training end.");
    }

    private void persistAutoencoderModel() {
        if (cfg.persist()) {
            for (int i = 0; i < layers.size(); i++) {
                layers.get(i).persist(cfg.workDir() + "/" + name + "-" + i);
            }
        }
    }

    private void copyModelFromEncoderToDecoder() {
        for (int i = 0; i <= layers.size() / 2; i++) {
            SRBMMapReduceJSA baseLayer = layers.get(i);
            SRBMMapReduceJSA mirroredLayer = layers.get(layers.size() - i - 1);
            mirroredLayer.getLayer().W = baseLayer.getLayer().W.transpose();
            mirroredLayer.getLayer().vbias = baseLayer.getLayer().hbias;
            mirroredLayer.getLayer().hbias = baseLayer.getLayer().vbias;
        }
    }

    public List<SRBMMapReduceJSA> getLayers(){
        return layers;
    }

    public Matrix transform(Matrix sample) {
        return layers.get(layers.size()-1).eval(sample);
    }
}