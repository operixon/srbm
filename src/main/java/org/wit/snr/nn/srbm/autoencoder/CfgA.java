package org.wit.snr.nn.srbm.autoencoder;

public class CfgA  extends  CfgAutoencoder {

    public final int numberOfEpochs = 20;

    @Override
    public int numdims() {
        return autoEncoderVisibleLayerSize;
    }

    @Override
    public int numhid() {
        return autoEncoderHiddenLayerSize;
    }

    @Override
    public int numberOfEpochs() {
        return numberOfEpochs;
    }



}
