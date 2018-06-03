package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.Configuration;
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
    public Matrix getHidProbs(Matrix X) {
        List<List<Double>> hiddenUnitsProbs = X
                .getMatrixAsCollection()
                .stream()
                .map(this::computeAllUnitsProbabilitiesFromHiddenLayer)
                .collect(toList());
        Matrix hp = new Matrix2D(hiddenUnitsProbs);
        if (hp.getRowsNumber() != cfg.numhid || hp.getColumnsNumber() != cfg.batchSize) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.numhid, cfg.batchSize, hp));
        }
        return hp;
    }

    /**
     * iterate on all hidden units and compute each hidden unit probability
     * using equation3
     *
     * @param sample visual layer units
     * @return list of probabilities hidden units to be in 1 state
     */
    private List<Double> computeAllUnitsProbabilitiesFromHiddenLayer(List<Double> sample) {
        return Stream.iterate(0, j -> j = j + 1)
                .limit(cfg.numhid)
                .map(j -> equation3.evaluate(j, sample))
                .collect(toList());
    }


}
