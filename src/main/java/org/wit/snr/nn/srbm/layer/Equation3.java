package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.math.ActivationFunction;

import java.io.Serializable;
import java.util.List;

public class Equation3 implements Serializable {

    final RbmCfg cfg;
    final Layer layer;
    final ActivationFunction activationFunction;

    public Equation3(RbmCfg cfg, Layer layer, ActivationFunction activationFunction) {
        this.cfg = cfg;
        this.layer = layer;
        this.activationFunction = activationFunction;
    }

    /**
     * P(hj|v) =g( λ /sigma^2(bj+sum_i wij*vi)).
     * if
     * z = (λ /sigma^2) * (bj+sum_i wij*vi)
     * then
     * P(hj|v) =g( z)
     * <pre>
     *     Hidden layer j-th unit probability for given visual input v
     * </pre>
     *
     * @param v visible layer data
     * @param j      hidden unit index
     * @return probability activation of single hidden unit with index j
     */
    public Double evaluate(int j, List<Double> v, double sigma) {
        final double cnst = (cfg.lambda() / (sigma * sigma)); // Obliczamy stałą część wyrarzenia

        final double z = cnst * (layer.getActivationSignalForHiddenUnit(v, j));
        return activationFunction.evaluate(z);
    }
}
