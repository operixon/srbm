package org.wit.snr.nn.dbn;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.SrbmMapReduce;
import org.wit.snr.nn.srbm.layer.Model;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Consumer;
import java.util.logging.Logger;

public class DbnAutoencoder {

    static Logger log = Logger.getLogger(DbnAutoencoder.class.getName());

    private final String name;
    private final RbmCfg cfg;
    private final int[] topology;
    private final List<SrbmMapReduce> layers = new LinkedList<SrbmMapReduce>();



    public DbnAutoencoder(String name, RbmCfg cfg, int[] topology) throws IllegalAccessException {

        if (topology.length < 3 || topology.length % 2 == 0) {
            throw new IllegalAccessException("Ten autoenkoder musi mieć minimum 3 warstwy i mieć sirodek symetrii");
        }
        this.name = name;
        this.cfg = cfg;
        this.topology = topology;
    }


    public void buildTopology() throws IOException, InterruptedException, CloneNotSupportedException {
        for (int i = 0; i < topology.length - 1; i++) {
            RbmCfg newLayerCfg = ((RbmCfg) cfg.clone()).numdims(topology[i])
                                                       .numhid(topology[i + 1])
                                                       .name(this.name + "-" + i);
            SrbmMapReduce newLayer = layers.size() == 0
                                        ? new SrbmMapReduce(newLayerCfg)
                                        : new SrbmMapReduce(layers.get(layers.size() - 1), newLayerCfg);
            layers.add(newLayer);
        }

    }

    public void load() throws IOException, ClassNotFoundException {
        if (cfg.load()) {
            for (int i = 0; i < layers.size(); i++) {
                layers.get(i).load(cfg.workDir() + "/" + name + "-" + i);
            }
        }
    }


    public void fit(List<List<Double>> x) {
        // na poczatek uczymy tylko pierwszą połowę enkodera
        for (int i = 0; i < layers.size() / 2; i++) {
            layers.get(i).train(x);
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
            SrbmMapReduce baseLayer = layers.get(i);
            SrbmMapReduce mirroredLayer = layers.get(layers.size() - i - 1);
            mirroredLayer.getLayer().W = baseLayer.getLayer().W.transpose();
            mirroredLayer.getLayer().vbias = baseLayer.getLayer().hbias;
            mirroredLayer.getLayer().hbias = baseLayer.getLayer().vbias;
        }
    }

    public List<SrbmMapReduce> getLayers() {
        return layers;
    }

    public Matrix transform(Matrix sample) {
        return layers.get(layers.size() - 1).eval(sample);
    }

    public void addHook(Consumer<Model> c) {
        layers.forEach(l->l.addHook(c));
    }
}
