package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.Configuration;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

public class NegativePhaseComputations {

    private final Equation2 equation2;
    private final Configuration cfg;

    public NegativePhaseComputations(Equation2 equation2, Configuration cfg) {
        this.equation2 = equation2;
        this.cfg = cfg;
    }

    /**
     * <pre>
     *
     * negdata := reconstruction of visible values given poshidstates (use Equation 2)
     * Iterate by N hidden samples
     *      For N-th hidden sample
     *              Iterate by all visible layer units
     *                  for i visible unit execute equation 2
     *
     * </pre>
     *
     * @param poshidstates matrix of hidden units states from positive phase
     * @return matrix of visible layers of reconstructed data without gibs sampling ( it is beater to use probabilities than boolean samples)
     */
    public Matrix getNegData(Matrix poshidstates, final double sigma) {
        List<List<Double>> visibleUnitsProbs = poshidstates
                .getMatrixAsCollection()
                .stream()
                .map( // For each hidden layer reproduce visual layer unit by unit using equation nr 2
                        hiddenLayerStates -> IntStream
                                .range(0, cfg.numdims())
                                .mapToDouble(visibleUnitIndex -> equation2.evaluate(visibleUnitIndex, hiddenLayerStates,sigma))
                                .boxed()
                                .collect(toList())
                )
                .collect(toList());
        Matrix hp = new Matrix2D(visibleUnitsProbs);
        if (hp.getRowsNumber() != cfg.numdims() || hp.getColumnsNumber() != cfg.batchSize()) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.batchSize(), cfg.numhid(), hp));
        }

        return hp;
    }
}
