package org.wit.snr.nn.srbm;

public class SigmoidFunction implements ActivationFunction {

    public double evaluate(double z) {
        return 1 / (1 + Math.pow(Math.E, -z));
    }

}
