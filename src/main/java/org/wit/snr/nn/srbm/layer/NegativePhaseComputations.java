package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.RbmCfg;

import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

public class NegativePhaseComputations {

    private final Equation2 equation2;
    private final RbmCfg cfg;

    public NegativePhaseComputations(Equation2 equation2, RbmCfg cfg) {
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
    public List<Double> getNegData(List<Double> hiddenLayerStates, final double sigma) {
        // For each hidden layer reproduce visual layer unit by unit using equation nr 2
        return IntStream.range(0, cfg.numdims())
                        .mapToDouble(visibleUnitIndex -> equation2.evaluate(visibleUnitIndex, hiddenLayerStates, sigma))
                        .boxed()
                        .collect(toList());
    }
}
