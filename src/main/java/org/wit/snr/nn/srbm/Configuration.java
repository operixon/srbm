/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

public class Configuration {

    private int numdims = 784; //# of visible units
    private int numhid = 784; //# of hidden units
    private double alpha = 0.005; // Learning rate, recomended value is 0.01
    private int batchSize = 20;
    // Dodaane zmienne
    private double mi = 0;
    private double lambda = 1; // Zgodnie z dokumentem zawsze ustawione na 1
    private int numberOfEpochs = 5;
    private double sparsneseFactor = 0.5; // Wspolczynnik P regulojacy rzadkość reprezentacji (7.2)
    private double sigmaInit = 0.5;
    private double sigmaDecay = 0.99;
    private String visualizationOutDirectory = "/home/artur/srbm";
    private boolean saveVisualization = false;
    private double acceptedError = 0.03;
    private boolean showVisualizationWindow = false;

    public boolean showVisualizationWindow() {
        return showVisualizationWindow;
    }

    public Configuration setShowVisualizationWindow(boolean showVisualizationWindow) {
        this.showVisualizationWindow = showVisualizationWindow;
        return this;
    }

    public Configuration setAcceptedError(double acceptedError) {
        this.acceptedError = acceptedError;
        return this;
    }

    public double acceptedError() {
        return acceptedError;
    }

    public Configuration setNumdims(int numdims) {
        this.numdims = numdims;
        return this;
    }

    public Configuration setNumhid(int numhid) {
        this.numhid = numhid;
        return this;
    }

    public Configuration setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public Configuration setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public Configuration setMi(double mi) {
        this.mi = mi;
        return this;
    }

    public Configuration setLambda(double lambda) {
        this.lambda = lambda;
        return this;
    }

    public Configuration setNumberOfEpochs(int numberOfEpochs) {
        this.numberOfEpochs = numberOfEpochs;
        return this;
    }

    public Configuration setSparsneseFactor(double sparsneseFactor) {
        this.sparsneseFactor = sparsneseFactor;
        return this;
    }

    public Configuration setSigmaInit(double sigmaInit) {
        this.sigmaInit = sigmaInit;
        return this;
    }

    public Configuration setSigmaDecay(double sigmaDecay) {
        this.sigmaDecay = sigmaDecay;
        return this;
    }

    public Configuration setVisualizationOutDirectory(String visualizationOutDirectory) {
        this.visualizationOutDirectory = visualizationOutDirectory;
        return this;
    }

    public Configuration setSaveVisualization(boolean saveVisualization) {
        this.saveVisualization = saveVisualization;
        return this;
    }

    public int numdims() {
        return numdims;
    }

    public int numhid() {
        return numhid;
    }

    public double alpha() {
        return alpha;
    }

    public int batchSize() {
        return batchSize;
    }

    public double mi() {
        return mi;
    }

    public double lambda() {
        return lambda;
    }

    public int numberOfEpochs() {
        return numberOfEpochs;
    }

    public double sparsneseFactor() {
        return sparsneseFactor;
    }

    public double sigmaInit() {
        return sigmaInit;
    }

    public double sigmaDecay() {
        return sigmaDecay;
    }

    public String visualizationOutDirectory() {
        return visualizationOutDirectory;
    }

    public boolean saveVisualization() {
        return saveVisualization;
    }

    public boolean visualizationWindow() {
        return showVisualizationWindow;
    }
}
