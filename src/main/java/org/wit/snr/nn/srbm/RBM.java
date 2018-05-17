/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.ActivationFunction;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;
import org.wit.snr.nn.srbm.math.function.GausianDensityFunction;
import org.wit.snr.nn.srbm.math.function.SigmoidFunction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * @author koperix
 */
public class RBM {

    static final Random random = new Random();
    static final Configuration cfg = new Configuration();
    final double sigma = 0.5;
    int currentEpoch = 0;
    final Layer layer = new Layer(cfg.numdims, cfg.numhid);
    final ActivationFunction gausianDensityFunction;
    final ActivationFunction sigmoidFunction;
    final TrainingSet trainingSet;

    public RBM() throws IOException {
        gausianDensityFunction = new GausianDensityFunction(sigma, cfg.mi);
        sigmoidFunction = new SigmoidFunction();
        trainingSet = new TrainingSetMinst();
    }

    public void train() {
        while (isConverged()) { // while (not converged)
            // for each training batch X batchSize x numdims
            // (randomly sample batchSize patches from data w/o replacement)
            for (int batchIdx = 0; batchIdx < 10; batchIdx++) {
                Matrix X = trainingSet.getTrainingBatch(cfg.batchSize);
                Matrix poshidprobs = getPosHidProbs(X);
                Matrix poshidstates = getPosHidStates(poshidprobs);
                Matrix negdata = getNegData(poshidstates);
                Matrix neghidprobs = getNegHidProbs(negdata);
                updateWeights(X, poshidprobs, negdata, neghidprobs);
                //vbias := vbias + α(rowsum(X) – rowsum(negdata))/batchSize
                updateVBias(X, negdata);
                //error := SquaredDiff(X,negdata)
                getError(X, negdata);
            }
            //update hbias (use Equation 6)
            updateHBias();
            // if (σ>0.05) σ := σ*0.99 end if
            currentEpoch++;
        }//#while end
    }//#train_rbm

    private boolean isConverged() {
        return currentEpoch < cfg.numberOfEpochs;
    }

    private void updateHBias() {
        throw new UnsupportedOperationException();
    }

    private void getError(Matrix x, Matrix negdata) {
        throw new UnsupportedOperationException();
    }

    /**
     * vbias := vbias + α(rowsum(X) – rowsum(negdata))/batchSize
     *
     * @param X
     * @param negdata
     */
    private void updateVBias(Matrix X, Matrix negdata) {
        Matrix rowsum_X = X.rowsum();
        Matrix rowsum_negdata = negdata.rowsum();
        Matrix biasDelta = rowsum_X.subtract(rowsum_negdata).scalarMultiply(cfg.alpha / (double) cfg.batchSize);
        layer.vbias.matrixAdd(biasDelta)
    }

    /**
     * // W := W + α(X*poshidprobsT – negdata*neghidprobsT)/batchSize
     *
     * @param X visible layer samples batch
     * @param poshidprobs positive phase hidden layer probabilities batch
     * @param negdata visible layer batch data from negative phase
     * @param neghidprobs  hidden layer probabilities for negative phase
     */
    private void updateWeights(Matrix X, Matrix poshidprobs, Matrix negdata, Matrix neghidprobs) {
        Matrix X_poshidprobsT = X.multiplication(poshidprobs.transpose()); //X*poshidprobsT
        Matrix negdata_neghidprobsT = negdata.multiplication(neghidprobs.transpose()); // negdata*neghidprobsT
        Matrix delta = X_poshidprobsT.subtract(negdata_neghidprobsT).scalarMultiply(cfg.alpha / (double) cfg.batchSize);
        layer.W = layer.W.matrixAdd(delta);
    }

    /**
     * neghidprobs := hidden unit probabilities given negdata (use Equation 3)
     *
     * @param negdata
     * @return
     */
    private Matrix getNegHidProbs(Matrix negdata) {
        return getPosHidProbs(negdata);
    }

    /**
     * negdata := reconstruction of visible values given poshidstates (use Equation 2)
     *
     * @param poshidstates
     * @return
     */
    private Matrix getNegData(Matrix poshidstates) {
        List<List<Double>> visibleUnitsProbs = poshidstates
                .getMatrixAsCollection()
                .stream()
                .map(sample -> equation2(sample))
                .collect(Collectors.toList());
        Matrix hp = new Matrix2D(visibleUnitsProbs);
        if (hp.getRows() != cfg.batchSize || hp.getColumns() != cfg.numdims) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.batchSize, cfg.numhid, hp));
        }
        return hp.gibsSampling();
    }

    /**
     * poshidstates := sample using poshidprobs
     *
     * @param poshidprobs
     * @return
     */
    private Matrix getPosHidStates(Matrix poshidprobs) {
        return poshidprobs.gibsSampling();
    }

    /**
     * hidden unit probabilities given X (use Equation 3)
     *
     * @param x
     * @return
     */
    private Matrix getPosHidProbs(Matrix X) {
        List<List<Double>> hiddenUnitsProbs = X
                .getMatrixAsCollection()
                .stream()
                .map(sample -> equation3(sample))
                .collect(Collectors.toList());
        Matrix hp = new Matrix2D(hiddenUnitsProbs);
        if (hp.getRows() != cfg.batchSize || hp.getColumns() != cfg.numhid) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.batchSize, cfg.numhid, hp));
        }
        return hp;
    }


    /**
     * P(hi|v) =g(lambda/sigma^2(bj+sum_i wij*vi)).
     * <pre>
     *     Hidden layer probabilities (h_probs) for given visual input v
     * </pre>
     *
     * @param sample layer data
     * @return probabilities of hidden unit states
     */
    private List<Double> equation3(List<Double> sample) {
        List<Double> h_probs = new ArrayList<>(cfg.numhid); // Wynik, czyli wektor prawdopodobieństw że dany neuron warstwy ukrytej ma wartość 1
        final double cnst = (cfg.lambda / (sigma * sigma)); // Obliczamy stałą część wyrarzenia
        for (int j = 0; j < cfg.numhid; j++) { // iterujemy po neuronach warstwy ukrytej
            double z = cnst * (layer.getActivationSignalForHiddenUnit(sample, j));
            h_probs.set(j, sigmoidFunction.evaluate(z));
        }
        return h_probs;
    }


    /**
     * Equation 2 P(Vi|h)
     * <pre>
     *     P(Vi|H) = N( LAMBDA ( Ci + SUMj(WijHj)), SIGMA * SIGMA )
     *     where
     *     N() is the Gaussian density
     *     Ci is the bias vector value with index i
     *     W is the weight matrix vale with index i,j     *
     * </pre>
     *
     * @param hiddenUnitStates
     * @return Probabilities for visual units reconstruction
     */
    private List<Double> equation2(List<Double> hiddenUnitStates) {
        List<Double> negdataProbs = new ArrayList(cfg.numdims);
        for (int i = 0; i < cfg.numdims; i++) { // iterate over all visible units
            double x = cfg.lambda * layer.getActivationSignalForVisibleUnit(hiddenUnitStates, i);
            negdataProbs.set(i, gausianDensityFunction.evaluate(x));
        }
        return negdataProbs;

    }


}