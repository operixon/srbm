/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import java.util.*;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

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
    TrainingSet<BitSet> trainingSet = new TrainingSetMock_DIM20_BOOLEAN();
    public void train() {
        //# [W , hbias, vbias]  = train_rbm(data, W, hbias, vbias, σ, alpha)
        // TODO: Do wyjaśnienia. Kiedy uważamy że ta flaga jest true?
        // Liczba epok ?
        while (currentEpoch < cfg.numberOfEpochs) {
            //# for each training  batch XnumdimsxbatchSize 
            //# (randomly sample batchSize patches from data w / o replacement)
            List<Boolean>[] trainingBatch = trainingSet.getTrainingBatch(cfg.batchSize);
            for (List<Boolean> sample : trainingBatch) {
                //# poshidprobs := hidden unit probabilities given X (use Equation 3)
                double[] hiddenUnitProbabilities = equation3(layer, cfg, sample, sigma);
                //# poshidstates:= sample using poshidprobs
                List<Boolean> hiddenUnitStates = gibsSampling(hiddenUnitProbabilities);
                //# negdata:= reconstruction of visible values given poshidstates(use Equation 2)
                double[] negdataProbs = equation2(layer, cfg, hiddenUnitStates, sigma);
                //# neghidprobs:= hidden unit probabilities given negdata (use Equation 3)
                List<Boolean> negdataStates = gibsSampling(negdataProbs);
                double[] neghidprobs = equation3(layer, cfg, negdataStates, sigma);
                //# W:= W + α(X * poshidprobsT – negdata * neghidprobsT)/batchSize
                layer.W = computeWCorrection(layer.W, sample, poshidprobs, negdata, neghidprobs, );
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


    private double sigmoid(double t) {
        return 1 / (1 + Math.pow(Math.E, -t));
    }

    private double[] equation3(
            Layer layer,
            Configuration cfg,
            List<Boolean> sample,
            double sigma) {
        double[] h = new double[cfg.numhid]; // Wynik
        final double cnst = (cfg.alpha / (sigma * sigma)); // Obliczamy stałą część wyrarzenia
        for (int j = 0; j < cfg.numhid; j++) { // iterujemy po indeksach warstwy ukrytej
            double sumWijVi = 0; // wynik sumy iloczynu W V po i
            for (int i = 0; i < cfg.numdims; i++) { // iterujemy po i, i sumujemy
                sumWijVi += layer.W[i][j] * (sample.get(i) == true ? 1.0 : 0);
            }
            h[j] = sigmoid(cnst * (layer.hbias[j] + sumWijVi);
        }
    }


    private static List<Boolean> gibsSampling(double[] in) {

        return Arrays.stream(in)
                .mapToObj(d -> d > random.nextDouble())
                .collect(toList());
    }


    private static double[] equation2(
            Layer layer,
            Configuration cfg,
            List<Boolean> hiddenUnitStates,
            double sigma
    ) {
        double[] negdataProbs = new double[cfg.numdims]; // Wektor prawdopodobieństw aktywacji dla rekonstrukcji warstwy widocznej
        for (int i = 0; i < cfg.numdims; i++) { // iterujemy po wszystkich jednostkach warstwy widocznej
            double sumWijhj = 0; // element sumy wystepujacej w rownaniu
            for (int j = 0; j < cfg.numhid; j++) {
                sumWijhj += layer.W[i][j] * (hiddenUnitStates.get(j) ? 1.0 : 0.0);
            }
            negdataProbs[i] = gausianDensity(
                    cfg.lambda * (layer.vbias[i] + sumWijhj),
                    cfg.mi,
                    Math.pow(sigma, 2)
            );
        }
        return negdataProbs;

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
