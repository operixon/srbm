package org.wit.snr.nn.srbm.autoencoder;

import org.wit.snr.nn.srbm.Configuration;

public abstract class CfgAutoencoder extends Configuration {


    public final double alpha = 0.01; // Learning rate, recomended value is 0.01
    public final int batchSize = 20;
    public final double mi = 0;
    public final double lambda = 1; // Zgodnie z dokumentem zawsze ustawione na 1
    public final double sparsneseFactor = 0.02; // Wspolczynnik P regulojacy rzadkość reprezentacji (7.2)
    public final double sigmaInit = 0.5;
    public final double sigmaDecay = 0.99;
    public final String visualizationOutDirectory = "/home/artur/srbm";
    public final boolean saveVisualization = true;



    @Override
    public double alpha() {
        return alpha;
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    @Override
    public double mi() {
        return mi;
    }

    @Override
    public double lambda() {
        return lambda;
    }

    @Override
    public double sparsneseFactor() {
        return sparsneseFactor;
    }

    @Override
    public double sigmaInit() {
        return sigmaInit;
    }

    @Override
    public double sigmaDecay() {
        return sigmaDecay;
    }

    @Override
    public String visualizationOutDirectory() {
        return visualizationOutDirectory;
    }

    @Override
    public boolean saveVisualization() {
        return saveVisualization;
    }

}
