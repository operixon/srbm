/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * @author koperix
 */
public class RBM {

    static final Random random = new Random();
    static final Configuration cfg = new Configuration();
    final double sigma = 0.5;
    int currentEpoch = 0;
    final Layer layer = new Layer(cfg.numdims, cfg.numhid);
    final ActivationFunction gausianDensityFunction;
    final ActivationFunction sigmoidFunction;
    final TrainingSet<List<Boolean>> trainingSet;

    public RBM() throws IOException {
        gausianDensityFunction = new GausianDensityFunction(sigma, cfg.mi);
        sigmoidFunction = new SigmoidFunction();
        trainingSet = new TrainingSetMinst();
    }



    public void train() {
        while (currentEpoch < cfg.numberOfEpochs) {
            List<List<Boolean>> trainingBatch = trainingSet.getTrainingBatch(cfg.batchSize);// X
            for (List<Boolean> v : trainingBatch) {
                List<Boolean> h = getHiddenValues(v);
                List<Boolean> vv = getReconstructedVisualValues(h);
                List<Boolean> hh = getReconstructedHiddenValues(vv);
                Matrix weightsDelta = getWeightsDelta(new Vector(v), new Vector(h), new Vector(vv), new Vector(hh));
                layer.W.substract(weightsDelta);
                System.out.println(layer.W.toString());
            }
        }//#while end
    }//#train_rbm

    /**
     * <pre>
     * Contrastive divergence weights correction based on positive phase and negative phase data.
     * Given with equation :
     * <code>delta W:=  α(X * poshidprobsT – negdata * neghidprobsT)/batchSize</code>
     * <b>where</b>
     *      X is v param
     *      poshidprobsT is transposed h param
     *      negdata is vv param
     *      neghidprobsT is transposed param
     *      X * poshidprobsT is Outer product of X and poshidprobsT vectors
     * </pre>
     *
     * @param v  visible layer values vector
     * @param h  hidden layer values vector
     * @param vv reconstructed visible layer values vector
     * @param hh reconstructed hidden layer values vector
     * @return change of weights
     */
    private Matrix getWeightsDelta(Vector v, Vector h, Vector vv, Vector hh) {
        Matrix vhOuterProduct = v.multiplyByTransposedVector(h);
        Matrix vvhhOuterProduct = vv.multiplyByTransposedVector(hh);
        Matrix delta = vhOuterProduct.substract(vvhhOuterProduct);
        final double factor = cfg.alpha / cfg.batchSize;
        delta.scalarMultiply(factor);
        return delta;
    }


    /**
     * Compute hidden layer values from reconstructed visible layer.
     * This is negative phase operation.
     * <pre>
     * <code>
     * neghidprobs := hidden unit probabilities given negdata (use Equation 3)
     * </code>
     * </pre>
     *
     * @param vv reconstructed visible layer units values
     * @return
     */
    private List<Boolean> getReconstructedHiddenValues(List<Boolean> vv) {
        ProbabilisticVector hiddenLayerProbs = equation3(vv);
        hiddenLayerProbs.sampling();
        return hiddenLayerProbs.getSampledValues();
    }

    // negdata := reconstruction of visible values given poshidstates (use Equation 2)
    private List<Boolean> getReconstructedVisualValues(List<Boolean> h) {
        ProbabilisticVector visualLayerReconstructionProbability = equation2(h);
        visualLayerReconstructionProbability.sampling();
        return visualLayerReconstructionProbability.getSampledValues();
    }

    /**
     * Compute values for hidden layer units.
     *
     * <pre>
     *
     * poshidprobs := hidden unit probabilities given X (use Equation 3)
     * poshidstates := sample using poshidprobs
     *
     * </pre>
     */
    private List<Boolean> getHiddenValues(List<Boolean> v) {
        ProbabilisticVector h_probs = equation3(v);
        h_probs.sampling();
        return h_probs.getSampledValues();
    }

    /**
     * P(hi|v) =g(lambda/sigma^2(bj+sum_i wij*vi)).
     * <pre>
     *     Hidden layer probabilities (h_probs) for given visual input v
     * </pre>
     *
     * @param v visible layer data
     * @return probabilities of hidden unit states
     */
    private ProbabilisticVector equation3(List<Boolean> v) {
        ProbabilisticVector h_probs = new ProbabilisticVector(cfg.numhid); // Wynik, czyli wektor prawdopodobieństw że dany neuron warstwy ukrytej ma wartość 1
        final double cnst = (cfg.lambda / (sigma * sigma)); // Obliczamy stałą część wyrarzenia
        for (int j = 0; j < cfg.numhid; j++) { // iterujemy po neuronach warstwy ukrytej
            double z = cnst * (layer.getActivationSignalForHiddenUnit(v, j));
            h_probs.getProbabilities().add(j, sigmoidFunction.evaluate(z));
        }
        return h_probs;
    }


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
     * @param hiddenUnitStates
     * @return Probabilities for visual units reconstruction
     */
    private ProbabilisticVector equation2(List<Boolean> hiddenUnitStates) {
        ProbabilisticVector negdataProbs = new ProbabilisticVector(cfg.numdims);
        for (int i = 0; i < cfg.numdims; i++) { // iterate over all visible units
            double x = cfg.lambda * layer.getActivationSignalForVisibleUnit(hiddenUnitStates, i);
            negdataProbs.getProbabilities().add(gausianDensityFunction.evaluate(x));
        }
        return negdataProbs;

    }


}
