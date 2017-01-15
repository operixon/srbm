/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import java.util.*;

/**
 *
 * @author koperix
 */
// TODO : Zaimplementować error
public class LearningAlgorithm {

    static final Random random = new Random();
    static final Configuration cfg = new Configuration();
    double sigma = 0.5;
    int currentEpoch = 0;
    Layer layer = new Layer(cfg.numdims,cfg.numhid);
    TrainingSet<Boolean> trainingSet = new TrainingSetMock_DIM20_BOOLEAN();

    public void train() {
        //# [W , hbias, vbias]  = train_rbm(data, W, hbias, vbias, σ, alpha)
        // TODO: Do wyjaśnienia. Kiedy uważamy że ta flaga jest true?
        // Liczba epok ?
        while (currentEpoch < cfg.numberOfEpochs) {
            //# for each training  batch XnumdimsxbatchSize 
            //# (randomly sample batchSize patches from data w / o replacement)
            List<Boolean>[] trainingBatch = trainingSet.getBatchOffRandomlySamples(cfg.batchSize);
            for (List<Boolean> sample : trainingBatch) {
                //# poshidprobs := hidden unit probabilities given X (use Equation 3)
                DoubleMatrix1D poshidprobs = computeHiddenLayerStatesProbabilitiesForGivenVisibleLayer(
                        cfg.numhid, sample, W, hbias, cfg.lambda, sigma, cfg.beta
                );
                //# poshidstates:= sample using poshidprobs
                DoubleMatrix1D poshidstates = computeStatesFromProbabilities(poshidprobs);
                //# negdata:= reconstruction of visible values given poshidstates(use Equation 2)
                DoubleMatrix1D negdata = computeVisibleLayerReconstructionForGivenHiddenLayerStates(
                        cfg.numdims,
                        cfg.numhid,
                        vbias.toArray(),
                        W.toArray(),
                        poshidstates.toArray(),
                        cfg.mi,
                        cfg.lambda,
                        sigma);
                //# neghidprobs:= hidden unit probabilities given negdata (use Equation 3)
                DoubleMatrix1D neghidprobs = computeHiddenLayerStatesProbabilitiesForGivenVisibleLayer(
                        cfg.numhid, negdata, W, hbias, cfg.lambda, sigma, cfg.beta
                );
                //# W:= W + α(X * poshidprobsT – negdata * neghidprobsT)/batchSize
                W = computeWCorrection(W, sample, poshidprobs, negdata, neghidprobs, cfg.alpha, cfg.batchSize);
                //# vbias:= vbias + alpha(rowsum(X) – rowsum(negdata) )/batchSize 
                vbias = computeVbiasCorrection(vbias, sample, negdata, cfg.alpha, cfg.batchSize);
                //# error := SquaredDiff(X, negdata)
                //TODO : error
            }//# end for
            //# update hbias  (use Equation  6) 
            hbias = updateHbias(hbias, cfg.learningRate, cfg.numsamples, trainingBatch);
            //# if (sigma > 0.05) sigma:= sigma * 0.99
            if (sigma > 0.05) {
                sigma = sigma * 0.99;
            }
            System.out.println(W);
            currentEpoch++;
        }//#while end  
    }//#train_rbm

    //TODO: validacja na bias->len do hsize, hsize do W itp
    //TODO: uzyć funkcyjnych elementów dla macierzy (append function)
    private static DoubleMatrix1D computeHiddenLayerStatesProbabilitiesForGivenVisibleLayer(
            int numhid,
            DoubleMatrix1D v,
            DoubleMatrix2D W,
            DoubleMatrix1D hbias,
            double lambda,
            double sigma,
            double beta
    ) {
        double[] posHidProbs = new double[numhid];
        for (int j = 0; j < posHidProbs.length; j++) {
            posHidProbs[j] = prob_Hj_v_eq3(hbias.get(j), W.toArray(), v.toArray(), j, lambda, sigma, beta);
        }
        return DoubleFactory1D.dense.make(posHidProbs);
    }

    private static double prob_Hj_v_eq3(double bj, double[][] W, double[] v, int j, double lambda, double sigma, double beta) {
        double sum_i = 0;
        for (int i = 0; i < v.length; i++) {
            sum_i = sum_i + W[i][j] * v[i];
        }
        return logisticFunction(
                (lambda / Math.pow(sigma, 2)) * (bj + sum_i), beta
        );
    }

    private static double logisticFunction(double x, double beta) {
        return 1 / (1 + Math.pow(Math.E, -2 * beta * x));
    }

    private static DoubleMatrix2D randMatrinx(int numdims, int numhid) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private static DoubleMatrix1D randnBooleanVector(int numdims) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private static DoubleMatrix1D computeStatesFromProbabilities(DoubleMatrix1D poshidprobs) {

        return poshidprobs.assign(new DoubleFunction() {
            @Override
            public double apply(double d) {
                return d > random.nextDouble() ? 1 : 0;
            }
        });

    }

    //TODO: use colt
    private static DoubleMatrix1D computeVisibleLayerReconstructionForGivenHiddenLayerStates(
            int numdims,
            int numhid,
            double[] vbias,
            double[][] W,
            double[] poshidstates,
            double mi,
            double lambda,
            double sigma
    ) {

        double[] negdataProbs = new double[numdims];

        for (int i = 0; i < negdataProbs.length; i++) {
            double sum_j = 0;
            for (int j = 0; j < numhid; j++) {
                sum_j = sum_j + W[i][j] * poshidstates[j];
            }
            negdataProbs[i] = gausianDensity(
                    lambda * (vbias[i] + sum_j),
                    mi,
                    Math.pow(sigma, 2)
            );
        }
        //TODO : Do wyjasnienia. Jest niezgodność w algorytmie. Równanie nr 2 ma sygnature Double -> Double
        // Tym czasem neg data to wartości zrekonstruowanego wejścia czyli powinny być 0,1
        // Zakładam że równanie zwraca prawdopodobieństwa które nalerzy potem przeliczyć na wartości.
        return computeStatesFromProbabilities(DoubleFactory1D.dense.make(negdataProbs));

    }

    private static double gausianDensity(double x, double mi, double sigma) {

        final double a = 1.0 / (sigma * Math.sqrt(2 * Math.PI));
        final double b = (-1L * Math.pow(x - mi, 2))
                / (2L * Math.pow(sigma, 2));
        return a * Math.pow(Math.E, b);
    }

    //TODO : Do wyjaśnienia. 
    // Z opisu algorytmu dla linijki W:= W + α(X * poshidprobsT – negdata * neghidprobsT)/batchSize
    // wynikało by że macierz wag nalezy zmodyfikowac dodając do kazdej wagi tą samą wartość
    // α(X * poshidprobsT – negdata * neghidprobsT)/batchSize. 
    // Ma to sens? Czy żle czytam notację.
    // Jeżeli (X * poshidprobsT) to iloczyn wektorowy to (X * poshidprobsT – negdata * neghidprobsT) jest wektorem
    // wówczas do macierzy W dodajemy wektor
    // w rzeciwnym wypadku do macierzy dodajemy liczbę (skalar)
    // ----------------------------------------------------------
    // W obecnej wersji zakładam że to iloczyn skalarny. (co jest troche bez sensu)
    private static DoubleMatrix2D computeWCorrection(
            DoubleMatrix2D W,
            final DoubleMatrix1D sample,
            final DoubleMatrix1D poshidprobs,
            final DoubleMatrix1D negdata,
            final DoubleMatrix1D neghidprobs,
            final double alpha,
            final int batchSize) {

        //W + alpha(X * poshidprobsT – negdata * neghidprobsT)/batchSize
        return W.assign(new DoubleFunction() {
            @Override
            public double apply(double d) {
                return d + alpha * (sample.zDotProduct(poshidprobs) - negdata.zDotProduct(neghidprobs)) / batchSize;
            }
        });
    }

    private static DoubleMatrix1D computeVbiasCorrection(
            final DoubleMatrix1D vbias,
            final DoubleMatrix1D sample,
            final DoubleMatrix1D negdata,
            final double alpha,
            final int batchSize
    ) {

        // vbias + alpha(rowsum(X) – rowsum(negdata) )/batchSize 
        return vbias.assign(new DoubleFunction() {
            @Override
            public double apply(double d) {
                return d + alpha * (sample.zSum() - negdata.zSum()) / batchSize;
            }
        });

    }

    private static DoubleMatrix1D updateHbias(DoubleMatrix1D hbias, double learningRate, int numsamples, DoubleMatrix1D[] batchOffRandomlySamples) {



    }

}
