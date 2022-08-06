package org.wit.snr.nn.srbm.math;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {

    double evaluate(double... x);

}
