package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.layer.Layer;
import org.wit.snr.nn.srbm.math.collection.Matrix;

public class datavis {

    public final Matrix X;
    public final int bathIdx;
    public final Layer layer;
    public final Matrix poshidprobs;
    public final Matrix poshidstates;
    public final Matrix negdata ;
    public final Matrix neghidprobs ;
    public final Matrix Wdelta ;
    public final Matrix vBiasDelta ;
    public final Matrix hBiasDelta;

    public datavis(Matrix x, int bathIdx, Layer layer, Matrix poshidprobs, Matrix poshidstates, Matrix negdata, Matrix neghidprobs, Matrix wdelta, Matrix vBiasDelta, Matrix hBiasDelta) {
        X = x;
        this.bathIdx = bathIdx;
        this.layer = layer;
        this.poshidprobs = poshidprobs;
        this.poshidstates = poshidstates;
        this.negdata = negdata;
        this.neghidprobs = neghidprobs;
        this.Wdelta = wdelta;
        this.vBiasDelta = vBiasDelta;
        this.hBiasDelta = hBiasDelta;
    }
}
