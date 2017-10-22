/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import java.util.*;

import static java.util.stream.Collectors.toList;

/**
 * @author koperix
 */
// TODO : Zaimplementować error
public class LearningAlgorithm {

    static final Random random = new Random();
    static final Configuration cfg = new Configuration();
    double sigma = 0.5;
    int currentEpoch = 0;
    Layer layer = new Layer(cfg.numdims, cfg.numhid);
    TrainingSet<List<Boolean>> trainingSet = new TrainingSetMock_DIM20_BOOLEAN();

    public void train() {


        //# [W , hbias, vbias]  = train_rbm(data, W, hbias, vbias, σ, alpha)
        // TODO : Wprowadzic warunek zbieznosci
        while (currentEpoch < cfg.numberOfEpochs) {
            //# for each training  batch XnumdimsxbatchSize 
            //# (randomly sample batchSize patches from data w / o replacement)
            List<List<Boolean>> trainingBatch = trainingSet.getTrainingBatch(cfg.batchSize);// X
            for (List<Boolean> sample : trainingBatch) {
                List<Double>  positivePhaseHiddenUnitProb   = equation3(layer, cfg, sample, sigma); //# poshidprobs := hidden unit probabilities given X (use Equation 3)
                List<Boolean> positivePhaseHiddenUnitState = gibsSampling(positivePhaseHiddenUnitProb); //# poshidstates:= sample using poshidprobs
                List<Double> reconstructionPhaseVisibleUnitProb = equation2(layer, cfg, positivePhaseHiddenUnitState, sigma);//# negdata:= reconstruction of visible values given poshidstates(use Equation 2)
                List<Boolean> reconstructionPhaseVisibleUnitState = gibsSampling(reconstructionPhaseVisibleUnitProb);//# neghidprobs:= hidden unit probabilities given negdata (use Equation 3)
                List<Double> reconstructionPhaseHiddenUnitProb = equation3(layer, cfg, reconstructionPhaseVisibleUnitState, sigma);
                //# W:= W + α(X * poshidprobsT – negdata * neghidprobsT)/batchSize
                List<List<Double>> newWeights = wCorrection(
                        layer.Wij,                           // W
                        cfg.alpha,                           // α
                        trainingBatch,                              // X === BLAD !!
                        // X jest macierza to sa wszystkie sample z paczki
                        positivePhaseHiddenUnitProb,        // poshidprobs
                        reconstructionPhaseVisibleUnitState, // negdata
                        reconstructionPhaseHiddenUnitProb,   // neghidprobs
                        cfg.batchSize);                      // bathSize
                //vbias:= vbias + alpha(rowsum(X) – rowsum(negdata) )/batchSize
                layer.updateVisualLayerBias();
                //# error := SquaredDiff(X, negdata)
                //TODO : error
            }//# end for
            //# update hbias  (use Equation  6) 
            //# if (sigma > 0.05) sigma:= sigma * 0.99
            if (sigma > 0.05) {
                sigma = sigma * 0.99;
            }
            // System.out.println(W);
            currentEpoch++;
        }//#while end  
    }//#train_rbm

    /**
     * <pre>
     * Contrastive divergence weights correction based on positive phase and negative phase data.
     * Given with equation :
     * <code>W:= W + α(X * poshidprobsT – negdata * neghidprobsT)/batchSize</code>
     * </pre>
     *
     * @param W           current weights between visible and hidden layer
     * @param alpha       const factor
     * @param X           visible unit states
     * @param poshidprobs possitive phase hidden unit prob
     * @param negdata     reconstruction phase visible unit state
     * @param neghidprobs reconstruction phase hidden unit prob
     * @param batchSize
     * @return corected weights matrix collection
     */
    private List<List<Double>> wCorrection(List<List<Double>> W,
                                           double alpha,
                                           List<List<Boolean>> X, // NUMDIMSxBATCHSIZE
                                           List<Double> poshidprobs,
                                           List<Boolean> negdata,
                                           List<Double> neghidprobs,
                                           int batchSize
    ) {
        Matrix wMatrix = new Matrix(W);
        Matrix X       = new Matrix(X);
        Matrix poshidprobsVector = new Matrix(Arrays.asList(poshidprobs));
        Matrix negdataVector = new Matrix(negdata);
        Matrix neghidprobsVector = new Matrix(neghidprobs);

        Matrix X_x_poshidprobsT = X.multiplyByTransposedVector(poshidprobsVector); // X * poshidprobsT
        Matrix negdata_x_neghidprobsT = negdataVector.multiplyByTransposedVector(neghidprobsVector); // negdata * neghidprobsT
        X_x_poshidprobsT
                .substract(negdata_x_neghidprobsT) // X * poshidprobsT – negdata * neghidprobsT
                .scalarDivide(batchSize) // (X * poshidprobsT – negdata * neghidprobsT)/batchSize
                .scalarMultiply(alpha); // α(X * poshidprobsT – negdata * neghidprobsT)/batchSize
        wMatrix.matrixAdd(X_x_poshidprobsT); // W + α(X * poshidprobsT – negdata * neghidprobsT)/batchSize


        return wMatrix.getData();

    }


    private double sigmoid(double z) {
        return 1 / (1 + Math.pow(Math.E, -z));
    }

    /**
     * P(hi|v) =g(lambda/sigma^2(bj+sum_i wij*vi)).
     * <pre>
     *     Probability of value eq 1 for hidden unit.
     * </pre>
     *
     * @param layer
     * @param cfg
     * @param sample
     * @param sigma
     * @return
     */
    private List<Double> equation3(
            Layer layer,
            Configuration cfg,
            List<Boolean> sample,
            double sigma) {
        List<Double> h = new ArrayList<Double>(cfg.numhid); // Wynik, czyli wektor prawdopodobieństw że dany neuron warstwy ukrytej ma wartość 1
        final double cnst = (cfg.lambda / (sigma * sigma)); // Obliczamy stałą część wyrarzenia
        for (int j = 0; j < cfg.numhid; j++) { // iterujemy po neuronach warstwy ukrytej
            double z = cnst * (layer.getActivationSignalForHiddenUnit(sample, j));
            h.add(j, sigmoid(z));
        }
        return h;
    }


    private static List<Boolean> gibsSampling(List<Double> in) {

        return in.stream()
                .map(d -> d > random.nextDouble())
                .collect(toList());
    }


    private static List<Double> equation2(
            Layer layer,
            Configuration cfg,
            List<Boolean> hiddenUnitStates,
            double sigma
    ) {
        List<Double> negdataProbs = new ArrayList<Double>(cfg.numdims); // Wektor prawdopodobieństw aktywacji dla rekonstrukcji warstwy widocznej
        for (int i = 0; i < cfg.numdims; i++) { // iterujemy po wszystkich jednostkach warstwy widocznej
            negdataProbs.add(
                    i,
                    gausianDensity(
                        cfg.lambda * layer.getActivationSignalForVisibleUnit(hiddenUnitStates,i),
                        cfg.mi,
                        Math.pow(sigma, 2)
            ));
        }
        return negdataProbs;

    }

    private static double gausianDensity(double x, double mi, double sigma) {

        final double a = 1.0 / (sigma * Math.sqrt(2 * Math.PI));
        final double b = (-1L * Math.pow(x - mi, 2))
                / (2L * Math.pow(sigma, 2));
        return a * Math.pow(Math.E, b);
    }




}
