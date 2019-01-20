/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

/**
 * @author koperix
 */
public abstract class Configuration {

    abstract public int numdims();

    abstract public int numhid();

    abstract public double alpha();

    abstract public int batchSize();

    abstract public double mi();

    abstract public double lambda();

    abstract public int numberOfEpochs();

    abstract public double sparsneseFactor();

    abstract public double sigmaInit();

    abstract public double sigmaDecay();

    abstract public String visualizationOutDirectory();

    abstract public boolean saveVisualization();
}
