package org.wit.snr.nn.srbm.autoencoder;

public class CfgA  extends  CfgAutoencoder {

    public final int numberOfEpochs = 60;

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
