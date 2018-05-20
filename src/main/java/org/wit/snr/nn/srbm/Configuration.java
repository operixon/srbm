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
class Configuration {
    
    // Orginalne stałe z algorytmu
    final int numdims = 784; //# of visible units
    final int numhid = 60; //# of hidden units
    final int numsamples = 100; //# of image patch samples
    final double alpha = 0.01;
    final double beta = 0.5;
    final int batchSize = 200;
 
    // Dodaane zmienne
    final double mi = 0.5;
    final double learningRate = 0.1;
    final double lambda = 1; // Zgodnie z dokumentem zawsze ustawione na 1
    final int numberOfEpochs = 1000;
    final double sparsneseFactor = 0.8; // Wspolczynnik P regulojacy rzadkość reprezentacji (7.2)

    final double sigma = 0.5;
}
