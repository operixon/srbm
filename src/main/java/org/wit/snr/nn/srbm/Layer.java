package org.wit.snr.nn.srbm;

import java.util.Arrays;

public class Layer {

    public final double[][] W;
    public final double[] vbias;
    public final double[] hbias;

    public Layer(int numdims, int numhid) {

        W = SRBMUtils.getRandomMatrix(numdims,numhid);

        vbias = new double[numdims];
        Arrays.fill(vbias,1);

        hbias = new double[numhid];
        Arrays.fill(hbias,1);
    }

}
