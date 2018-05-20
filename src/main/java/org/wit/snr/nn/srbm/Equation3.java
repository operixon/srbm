package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.ActivationFunction;

import java.util.List;

public class Equation3 {

    final Configuration cfg;
    final Layer layer;
    final ActivationFunction activationFunction;

    public Equation3(Configuration cfg, Layer layer, ActivationFunction activationFunction) {
        this.cfg = cfg;
        this.layer = layer;
        this.activationFunction = activationFunction;
    }

    /**
     * P(hj|v) =g( λ /sigma^2(bj+sum_i wij*vi)).
     * if
     * z = λ /sigma^2(bj+sum_i wij*vi)
     * then
     * P(hj|v) =g( z)
     * <pre>
     *     Hidden layer probabilities (h_probs) for given visual input v
     * </pre>
     *
     * @param sample visible layer data
     * @param j      hidden unit index
     * @return probability activation of single hidden unit with index j
     */
    public Double evaluate(int j, List<Double> sample) {
        final double cnst = (cfg.lambda / (cfg.sigma * cfg.sigma)); // Obliczamy stałą część wyrarzenia
        final double z = cnst * (layer.getActivationSignalForHiddenUnit(sample, j));
        return activationFunction.evaluate(z);
    }
}
