package org.wit.snr.nn.srbm.math.function;

import org.wit.snr.nn.srbm.math.ActivationFunction;

/**
 * Logistic function described by equation:
 * <pre>
 *
 *              1
 *  g(x) = ------------
 *          (1 + e^-x)
 *
 *  </pre>
 */
public class SigmoidFunction implements ActivationFunction {

    /**
     *
     * @param x function arguments table with size 1.
     * @return g(x) value
     */
    public double evaluate(double... x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x[0]));
    }

}
