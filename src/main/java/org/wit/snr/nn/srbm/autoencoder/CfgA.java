package org.wit.snr.nn.srbm.autoencoder;

public class CfgA  extends  CfgAutoencoder {

    public final int numdims = 784;
    public final int numhid = 196;
    public final int numberOfEpochs = 14;

    @Override
    public int numdims() {
        return numdims;
    }

    @Override
    public int numhid() {
        return numhid;
    }

    @Override
    public int numberOfEpochs() {
        return numberOfEpochs;
    }



}
