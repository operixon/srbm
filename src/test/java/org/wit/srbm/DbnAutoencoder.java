package org.wit.srbm;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.SRBMMapReduceJSA;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class DbnAutoencoder {
    private String name;
    private RbmCfg cfg;
    private int[] topology;
    private List<SRBMMapReduceJSA> layers = new LinkedList<SRBMMapReduceJSA>();



    public DbnAutoencoder(String name, RbmCfg cfg, int[] topology) throws IllegalAccessException {

        if(topology.length < 3 || topology.length%2 ==0){
            throw new IllegalAccessException("Ten autoenkoder musi mieć minimum 3 warstwy i mieć sirodek symetrii");
        }
        this.name = name;
        this.cfg = cfg;
        this.topology = topology;
    }


    public void buildTopology() throws IOException, InterruptedException, CloneNotSupportedException {

        // na poczatek uczymy tylko pierwszą połowę enkodera
        for (int i = 0; i <= (topology.length-1)/2; i++) {
            RbmCfg newLayerCfg = ((RbmCfg) cfg.clone()).numdims(topology[i])
                                                       .numhid(topology[i + 1])
                                                       .name(this.name + "-" + i);
            SRBMMapReduceJSA newLayer = layers.size() == 0
                            ? new SRBMMapReduceJSA(newLayerCfg)
                            : new SRBMMapReduceJSA(layers.get(layers.size()-1),newLayerCfg);
            layers.add(newLayer);
        }
    }


    public void fit() {

    }
}
