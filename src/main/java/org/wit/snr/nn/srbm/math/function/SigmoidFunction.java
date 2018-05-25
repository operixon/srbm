package org.wit.snr.nn.srbm.math.function;

import org.wit.snr.nn.srbm.math.ActivationFunction;

public class SigmoidFunction implements ActivationFunction {

    public double evaluate(double z) {
        return 1.0 / (1.0 + Math.pow(Math.E, -z));
    }

}
