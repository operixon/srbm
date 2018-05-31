package org.wit.snr.nn.srbm.math.function;

import org.wit.snr.nn.srbm.math.ActivationFunction;

/**
 * Gausian density function
 *
 * <pre>
 *
 *                  1                 -( x - μ )^2
 * N(x,σ) =  ---------------- exp ( ---------------- )
 *             σ ((2π)^0.5)              2σ^2
 *
 *
 * </pre>
 */
public class GausianDensityFunction implements ActivationFunction {

    final double mi;


    public GausianDensityFunction(double mi) {
        this.mi = mi;
    }

    final private static double SQRT_2_PI = Math.sqrt(2 * Math.PI);

    /**
     * @param Xn table with two arguments x[0] - x variable; x[1] sigma argument
     * @return N(x, σ ^ 2) function value for x[0] , x[1] params
     */
    public double evaluate(double... Xn) {
        final double x = Xn[0];
        //   System.out.println("x-->"+x);
        final double sigma = Xn[1];
        final double a = 1.0 / (sigma * SQRT_2_PI);
        final double b = (-1.0 * Math.pow(x - mi, 2)) / (2 * sigma * sigma);
        return a * Math.exp(b);
    }

}
