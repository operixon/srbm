package org.wit.snr.nn.srbm.autoencoder;

public class CfgB extends CfgAutoencoder {

    public final int numberOfEpochs = 60;

    @Override
    public int numdims() {
        return autoEncoderHiddenLayerSize;
    }

    @Override
    public int numhid() {
        return autoEncoderVisibleLayerSize;
    }

    @Override
    public int numberOfEpochs() {
        return numberOfEpochs;
    }

}
