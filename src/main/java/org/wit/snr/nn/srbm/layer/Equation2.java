package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.RbmCfg;
import org.wit.snr.nn.srbm.math.ActivationFunction;

import java.util.List;

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
 * @return Probabilities for visual units reconstruction
 */
public class Equation2 {

    final RbmCfg cfg;
    final Layer layer;
    final ActivationFunction activationFunction;

    public Equation2(RbmCfg cfg, Layer layer, ActivationFunction activationFunction) {
        this.cfg = cfg;
        this.layer = layer;
        this.activationFunction = activationFunction;
    }

    /**
     * <pre>
     *
     * P(Vi|H) = N( LAMBDA ( Ci + SUMj(WijHj)), SIGMA * SIGMA )
     *
     * </pre>
     *
     * @param i                index visible layer unit to evaluate data reconstruction
     * @param hiddenUnitStates hidden layer states
     * @return negdata for single unit in visual layer
     */
    public Double evaluate(final int i, final List<Double> hiddenUnitStates, final double sigma) {
        double x = cfg.lambda() * layer.getActivationSignalForVisibleUnit(hiddenUnitStates, i);
        return activationFunction.evaluate(x, sigma);
    }
}
