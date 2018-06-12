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
public class Configuration {
    
    // Orginalne stałe z algorytmu
    public final int numdims = 784; //# of visible units
    public final int numhid = 200; //# of hidden units
    public final double alpha = 0.01; // Learning rate, recomended value is 0.01
    public final int batchSize = 200;
 
    // Dodaane zmienne
    public final double mi = 0;
    public final double lambda = 1; // Zgodnie z dokumentem zawsze ustawione na 1
    public final int numberOfEpochs = 60;
    public final double sparsneseFactor = 0.03; // Wspolczynnik P regulojacy rzadkość reprezentacji (7.2)

    public double sigma = 0.5;
    public final String visualizationOutDirectory = "/srbm";
    public final boolean saveVisualization = true;
}
