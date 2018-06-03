package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;

public class MiniBatchTrainingResult {

    final Matrix W;
    final Matrix vbias;
    final Matrix hbias;

    public MiniBatchTrainingResult(Matrix w, Matrix vbias, Matrix hbias) {
        W = w;
        this.vbias = vbias;
        this.hbias = hbias;
    }

    public Matrix getW() {
        return W;
    }

    public Matrix getVbias() {
        return vbias;
    }

    public Matrix getHbias() {
        return hbias;
    }
}
