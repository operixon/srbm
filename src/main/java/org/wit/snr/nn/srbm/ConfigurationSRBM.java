/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

/**
 *
 * @author koperix
 */
public class ConfigurationSRBM extends Configuration {

    // Orginalne stałe z algorytmu
    public final int numdims = 784; //# of visible units
    public final int numhid = 784; //# of hidden units
    public final double alpha = 0.01; // Learning rate, recomended value is 0.01
    public final int batchSize = 200;

    // Dodaane zmienne
    public final double mi = 0;
    public final double lambda = 1; // Zgodnie z dokumentem zawsze ustawione na 1
    public final int numberOfEpochs = 600000000;
    public final double sparsneseFactor = 0.05; // Wspolczynnik P regulojacy rzadkość reprezentacji (7.2)

    public final double sigmaInit = 0.5;
    public final double sigmaDecay = 0.99;
    public final String visualizationOutDirectory = "/home/artur/srbm";
    public final boolean saveVisualization = true;


    @Override
    public int numdims() {
        return numdims;
    }

    @Override
    public int numhid() {
        return numhid;
    }

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
    public int numberOfEpochs() {
        return numberOfEpochs;
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
