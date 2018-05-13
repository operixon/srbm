package org.wit.snr.nn.srbm;

public class GausianDensityFunction implements ActivationFunction {

    final double sigma;
    final double mi;

    public GausianDensityFunction(double sigma, double mi) {
        this.sigma = sigma;
        this.mi = mi;
    }

    public double evaluate(double x) {

        final double a = 1.0 / (sigma * Math.sqrt(2 * Math.PI));
        final double b = (-1L * Math.pow(x - mi, 2))
                / (2L * Math.pow(sigma, 2));
        return a * Math.pow(Math.E, b);
    }
}
