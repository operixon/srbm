package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.math.RandomSampler;

import java.util.List;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

public class PositivePhaseComputations {

    private final Equation3 equation3;
    private final RbmCfg cfg;
    private RandomSampler sampler = new RandomSampler();

    public PositivePhaseComputations(Equation3 equation3, RbmCfg cfg) {
        this.equation3 = equation3;
        this.cfg = cfg;
    }

    /**
     * poshidstates := sample using poshidprobs
     *
     * @param poshidprobs
     * @return
     */
    public List<Double> getHidStates(List<Double> poshidprobs) {
        return sampler.sample(poshidprobs);
    }

    /**
     *
     * hidden unit probabilities given X (use Equation 3)
     *
     * @param X batch of samples
     * @return
     */
    public List<Double> getHidProbs(List<Double> sample,final double sigma) {
        List<Double> probs = computeAllUnitsProbabilitiesFromHiddenLayer(sample,sigma);
        if (probs.size() != cfg.numhid() ) {
            throw new IllegalStateException();
        }
        return probs;
    }

    /**
     * iterate on all hidden units and compute each hidden unit probability
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


}
