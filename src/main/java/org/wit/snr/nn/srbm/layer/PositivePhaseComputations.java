package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.Configuration;
import org.wit.snr.nn.srbm.math.ActivationFunction;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.util.List;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

public class PositivePhaseComputations {

    private final Equation3 equation3;
    private final Configuration cfg;

    public PositivePhaseComputations(Equation3 equation3, Configuration cfg) {
        this.equation3 = equation3;
        this.cfg = cfg;
    }

    /**
     * poshidstates := sample using poshidprobs
     *
     * @param poshidprobs
     * @return
     */
    public Matrix getHidStates(Matrix poshidprobs) {
        return poshidprobs.gibsSampling();
    }

    /**
     *
     * hidden unit probabilities given X (use Equation 3)
     *
     * @param X batch of samples
     * @return
     */
    public Matrix getHidProbs(Matrix X,final double sigma) {
        List<List<Double>> hiddenUnitsProbs = X
                .getMatrixAsCollection()
                .stream()
                .map((List<Double> sample) -> computeAllUnitsProbabilitiesFromHiddenLayer(sample,sigma))
                .collect(toList());
        Matrix hp = new Matrix2D(hiddenUnitsProbs);
        if (hp.getRowsNumber() != cfg.numhid() || hp.getColumnsNumber() != cfg.batchSize()) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.numhid(), cfg.batchSize(), hp));
        }
        return hp;
    }

    public Matrix getHidProbs2(Matrix X,final double sigma) {




        List<List<Double>> hiddenUnitsProbs = X
                .getMatrixAsCollection()
                .stream()
                .map((List<Double> sample) -> computeAllUnitsProbabilitiesFromHiddenLayer(sample,sigma))
                .collect(toList());
        Matrix hp = new Matrix2D(hiddenUnitsProbs);
        if (hp.getRowsNumber() != cfg.numhid() || hp.getColumnsNumber() != cfg.batchSize()) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.numhid(), cfg.batchSize(), hp));
        }
        return hp;
    }

    /**
     * iterate on all hidden units and propagate each hidden unit probability
     * using equation3
     *
     * @param sample visual layer units
     * @return list of probabilities hidden units to be in 1 state
     */
    private List<Double> computeAllUnitsProbabilitiesFromHiddenLayer(List<Double> sample, double sigma) {
        return Stream.iterate(0, j -> j = j + 1)
                .limit(cfg.numhid())
                .map(j -> equation3.evaluate(j, sample,sigma))
                .collect(toList());
    }

    private List<Double> computeAllUnitsProbabilitiesFromHiddenLayer2(List<Double> sample, double sigma) {

        Matrix v = Matrix2D.createColumnVector(sample);
        Matrix b = equation3.layer.hbias;
        Matrix w = equation3.layer.W;
        Matrix Ph = w.transpose().multiplication(v).matrixAdd(b);
        ActivationFunction g = equation3.activationFunction;
        final  double ls2 = cfg.lambda()/(sigma*sigma);
        return Ph.getDataAsList()
                 .stream()
                 .map( phj -> g.evaluate(ls2*phj))
                 .collect(toList());
    }


}
