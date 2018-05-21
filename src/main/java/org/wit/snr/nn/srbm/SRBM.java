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
import org.wit.snr.nn.srbm.trainingset.TrainingSetMinst;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

/**
 * @author koperix
 */
public class SRBM {

    final Configuration cfg = new Configuration();
    final Layer layer;
    final ActivationFunction gausianDensityFunction;
    final TrainingSet trainingSet;
    final HiddenLayerComputations hiddenLayerComputations;
    final HiddenBiasAdaptation hiddenBiasAdaptation;
    int currentEpoch;

    public SRBM() throws IOException {
        this.layer = new Layer(cfg.numdims, cfg.numhid);
        gausianDensityFunction = new GausianDensityFunction(cfg.sigma, cfg.mi);
        trainingSet = new TrainingSetMinst();
        Equation3 equation3 = new Equation3(cfg, layer, new SigmoidFunction());
        hiddenLayerComputations = new HiddenLayerComputations(equation3, cfg);
        hiddenBiasAdaptation = new HiddenBiasAdaptation(equation3);
    }

    public void train() {
        currentEpoch = 0;
        while (isConverged()) {

            for (int batchIdx = 0; batchIdx < 10; batchIdx++) {
                final long a = System.currentTimeMillis();

                Matrix X = trainingSet.getTrainingBatch(cfg.batchSize);
                final double trainBatchTs = System.currentTimeMillis();

                Matrix poshidprobs = hiddenLayerComputations.getHidProbs(X);
                Matrix poshidstates = hiddenLayerComputations.getHidStates(poshidprobs);
                double positiveTs = System.currentTimeMillis();

                Matrix negdata = getNegData(poshidstates);
                Matrix neghidprobs = getNegHidProbs(negdata);
                double negTs = System.currentTimeMillis();

                updateWeights(X, poshidprobs, negdata, neghidprobs);
                double wUpdateTs = System.currentTimeMillis();

                updateVBias(X, negdata);
                double vbiasTime = System.currentTimeMillis();

                updateError(X, negdata);
                double errorTime = System.currentTimeMillis();

                updateHBias(X);
                double hbiasTime = System.currentTimeMillis();

                System.out.println("=<< time=" + ((System.currentTimeMillis() - a) / 1000.0) + "s, epoch=" + currentEpoch + " error=" + layer.error);
                System.out.printf(
                        "| batch gen %.3fs " +
                                "| positiv %.3fs " +
                                "| negative %.3fs " +
                                "| W  %.3fs " +
                                "| vbias  %.3fs " +
                                "| error %.3fs " +
                                "| hbias %.3fs" + "%n",
                        (trainBatchTs - a) / 1000.0,
                        (positiveTs - trainBatchTs) / 1000.0,
                        (negTs - positiveTs) / 1000.0,
                        (wUpdateTs - negTs) / 1000.0,
                        (vbiasTime - wUpdateTs) / 1000.0,
                        (errorTime - vbiasTime) / 1000.0,
                        (hbiasTime - errorTime) / 1000.0);
                currentEpoch++;
            }
            // Zgodnie z algorytmem
            // update hbias (use Equation 6)
            // powinno być w tym miejscu ale wtedy nie mam dostępu do
            // paczki trenującej

            if (cfg.sigma > 0.05) cfg.sigma = cfg.sigma * 0.99;

        }//#while end
    }//#train_rbm

    private boolean isConverged() {
        return currentEpoch < cfg.numberOfEpochs;
    }

    private void updateHBias(Matrix X) {
        List<Double> updatedBiasData = Stream
                .iterate(0, j -> j = j + 1)
                .limit(cfg.numhid)
                .parallel()
                .map(j ->
                        hiddenBiasAdaptation.getHiddenBiasUnit(
                                layer.vbias.get(j, 0),
                                cfg.learningRate,
                                cfg.batchSize,
                                j,
                                cfg.sparsneseFactor,
                                X))
                .collect(toList());
        layer.hbias = Matrix2D.createColumnVector(updatedBiasData);
    }


    /**
     * error := SquaredDiff(X,negdata)
     *
     * @param X
     * @param negdata
     */
    private void updateError(Matrix X, Matrix negdata) {

        List<List<Double>> dataBatch = X.getMatrixAsCollection();
        List<List<Double>> negDataBatch = negdata.getMatrixAsCollection();
        if (dataBatch.size() != negDataBatch.size() || dataBatch.get(0).size() != negDataBatch.get(0).size()) {
            throw new IllegalStateException(String.format("x.size=%d; xx.size=%d; x.get(0).size=%d; xx.get(0).size=%d", dataBatch.size(), negDataBatch.size(), dataBatch.get(0).size(), negDataBatch.get(0).size()));
        }
        // mse = 1/n ( sum (n , i = 1, (Yi - Y'i)^2)
        // Java stream not provides zip api to glue two collections
        // so it must be done in for() fashion way
        double error = 0;
        for (int sampleIdx = 0; sampleIdx < dataBatch.size(); sampleIdx++) {
            List<Double> data = dataBatch.get(sampleIdx);
            List<Double> negData = negDataBatch.get(sampleIdx);
            for (int unitIdx = 0; unitIdx < dataBatch.get(0).size(); unitIdx++) {
                double unit = data.get(unitIdx);
                double negUnit = negData.get(unitIdx);
                error += (unit - negUnit) * (unit - negUnit);
            }
        }
        layer.error = error / (cfg.batchSize * cfg.numdims);
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
        layer.vbias = layer.vbias.matrixAdd(biasDelta);
    }

    /**
     * W := W + α(X*poshidprobsT – negdata*neghidprobsT)/batchSize
     *
     * @param X           visible layer samples batch
     * @param poshidprobs positive phase hidden layer probabilities batch
     * @param negdata     visible layer batch data from negative phase
     * @param neghidprobs hidden layer probabilities for negative phase
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
        return hiddenLayerComputations.getHidProbs(negdata);
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
                .parallel()
                .map(sample -> equation2(sample))
                .collect(toList());
        Matrix hp = new Matrix2D(visibleUnitsProbs);
        if (hp.getRowsNumber() != cfg.numdims || hp.getColumnsNumber() != cfg.batchSize) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.batchSize, cfg.numhid, hp));
        }
        return hp.gibsSampling();
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
            negdataProbs.add(gausianDensityFunction.evaluate(x));
        }
        return negdataProbs;

    }


}
