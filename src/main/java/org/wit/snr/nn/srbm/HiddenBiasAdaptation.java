package org.wit.snr.nn.srbm;


/**
 * <h1>Sparse deep belief net models for visual area V2</h1>
 *
 * <B>Chaitanya Ekanadham</B>
 *
 * <h2>7.2 Hidden bias adaptation</h2>
 *
 * <pre>
 * To satisfy the constraint described in Equation 5, we first calculate for each hidden unit
 * the average probability of firing over the data. Subtracting the parameter p (see Section 3.1) from
 * this value yields the update rule:
 *
 * <code>hj := hj - η (1/m * ∑(from i=1, to m, E[hj(i)|v(i)])-p)</code>          ( Equation 6 )
 *
 * where η is the learning rate, m is the number of samples, and v(i) denotes the i’th image sample.
 * E(hj(i)|v(i)) is calculated using Equation 3. In practice, we performed only one update step on each
 * epoch, so the average hidden unit activations were not necessarily p in the leading few training
 * epochs. We found that in practice, performing a single gradient step on each epoch seemed to
 * work better than performing multiple gradient steps on each epoch so that the activations were
 * exactly p. This may be due to sensitivity of the constrained optimization space to the
 * initialization of the parameters.
 *
 * </pre>
 */

public class HiddenBiasAdaptation {

    //hj := hj - ni (1/m * sum(from i=1, to m, E[hj(i)|v(i)])-p)

    public void updateBias() {

    }

}
