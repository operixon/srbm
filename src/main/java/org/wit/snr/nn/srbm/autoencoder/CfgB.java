package org.wit.snr.nn.srbm.autoencoder;

public class CfgB extends CfgAutoencoder {

    public final int numdims = 196;
    public final int numhid =  784;
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
