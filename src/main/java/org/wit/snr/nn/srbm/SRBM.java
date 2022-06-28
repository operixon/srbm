/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.layer.*;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;
import org.wit.snr.nn.srbm.math.function.SigmoidFunction;
import org.wit.snr.nn.srbm.monitoring.Timer;
import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

/**
 * @author koperix
 */
public abstract class SRBM {

    final RbmCfg cfg;
    Layer layer;
    final PositivePhaseComputations positivePhaseComputations;
    final NegativePhaseComputations negativePhaseComputations;
    final HiddenBiasAdaptation hiddenBiasAdaptation;

    final ThreadLocal<Timer> timer = ThreadLocal.withInitial(Timer::new);
    final AtomicInteger currentEpoch = new AtomicInteger(0);
    final AtomicInteger miniBatchIndex = new AtomicInteger(0);


    final String sessionId = "srbm-" + System.currentTimeMillis();

    protected double sigma;

    public abstract void train(List<List<Double>> x);

    public abstract SRBM setNext(SRBM next);

    public SRBM(RbmCfg cfg) throws IOException, InterruptedException {
        this.cfg = cfg;
        sigma = cfg.sigmaInit();
        this.layer = new Layer(this.cfg.numdims(), this.cfg.numhid());
        Equation3 equation3 = new Equation3(this.cfg, layer, new SigmoidFunction());
        positivePhaseComputations = new PositivePhaseComputations(equation3, this.cfg);
        hiddenBiasAdaptation = new HiddenBiasAdaptation(equation3);
        negativePhaseComputations = new NegativePhaseComputations(
                new Equation2(this.cfg, layer, new SigmoidFunction()),
                this.cfg
        );
    }


    protected Matrix getNegData(Matrix poshidstates) {
        Matrix negData = negativePhaseComputations.getNegData(poshidstates, sigma);
        timer.get().mark("negdata");
        return negData;
    }



    protected Matrix getHidStates(Matrix poshidprobs) {
        Matrix hidStates = positivePhaseComputations.getHidStates(poshidprobs);
        timer.get().mark("hidstates");
        return hidStates;
    }

    protected Matrix getHidProbs(Matrix X) {
        Matrix hidProbs = positivePhaseComputations.getHidProbs(X, sigma);
        timer.get().mark("hidprobs");
        return hidProbs;
    }

    /**
     * Oblicza wektor bias ukryty poprzez wywołanie równania nr 3 dla każdej jednostki wektora
     */
    protected Matrix getHBiasDelta(Matrix X) {
        List<Double> hBiasDelta = new LinkedList<>();
        for (int j = 0; j < cfg.numhid(); j++) {
            Double hiddenBiasUnitDelta = hiddenBiasAdaptation.getHiddenBiasUnitDelta(
                    cfg.alpha(),
                    cfg.batchSize(),
                    j,
                    cfg.sparsneseFactor(),
                    X,
                    sigma
            );
            hBiasDelta.add(hiddenBiasUnitDelta);
        }
        timer.get().mark("hbias");
        return Matrix2D.createColumnVector(hBiasDelta);
    }


    /**
     * error := SquaredDiff(X,negdata)
     *
     * @param X
     * @param negdata
     */
    protected void updateError(Matrix X, Matrix negdata) {

        List<List<Double>> dataBatch = X.getMatrixAsCollection();
        List<List<Double>> negDataBatch = negdata.getMatrixAsCollection();
        if (dataBatch.size() != negDataBatch.size() || dataBatch.get(0).size() != negDataBatch.get(0).size()) {
            throw new IllegalStateException(String.format("x.size=%d; xx.size=%d; x.get(0).size=%d; xx.get(0).size=%d", dataBatch.size(), negDataBatch.size(), dataBatch.get(0).size(), negDataBatch.get(0).size()));
        }
        // mse = 1/n ( sum (n , i = 1, (Yi - Y'i)^2)
        // Java stream not provides zip api to glue two collections
        // so it must be done in for() fashion way
        double error = 0;
        for (int sampleIdx = 0; sampleIdx < dataBatch.size(); sampleIdx++) {
            List<Double> data = dataBatch.get(sampleIdx);
            List<Double> negData = negDataBatch.get(sampleIdx);
            for (int unitIdx = 0; unitIdx < dataBatch.get(0).size(); unitIdx++) {
                double unit = data.get(unitIdx);
                double negUnit = negData.get(unitIdx);
                error += (unit - negUnit) * (unit - negUnit);
            }
        }
        layer.error = error / (cfg.batchSize() * cfg.numdims());
        timer.get().mark("error");
    }

    /**
     * vbias := vbias + α(rowsum(X) – rowsum(negdata))/batchSize
     *
     * @param X
     * @param negdata
     */
    protected Matrix updateVBias(Matrix X, Matrix negdata) {
        Matrix rowsum_X = X.rowsum();
        Matrix rowsum_negdata = negdata.rowsum();
        Matrix biasDelta = rowsum_X.subtract(rowsum_negdata).scalarMultiply(cfg.alpha() / (double) cfg.batchSize());
        timer.get().mark("vbias");
        return biasDelta;
    }

    /**
     * W := W + α(X*poshidprobsT – negdata*neghidprobsT)/batchSize
     *
     * @param X           visible layer samples batch
     * @param poshidprobs positive phase hidden layer probabilities batch
     * @param negdata     visible layer batch data from negative phase
     * @param neghidprobs hidden layer probabilities for negative phase
     */
    protected Matrix updateWeights(
            final Matrix X,
            final Matrix poshidprobs,
            final Matrix negdata,
            final Matrix neghidprobs) {

        Matrix x_poshidprobsT = X.multiplication(poshidprobs.transpose()); // X*poshidprobsT
        Matrix negdata_neghidprobsT = negdata.multiplication(neghidprobs.transpose());// negdata*neghidprobsT
        Matrix x_poshidprobsT_negdata_neghidprobsT = x_poshidprobsT.subtract(negdata_neghidprobsT);// (X*poshidprobsT – negdata*neghidprobsT)
        Matrix delta = x_poshidprobsT_negdata_neghidprobsT.scalarDivide((double) cfg.batchSize()).scalarMultiply(cfg.alpha());// a*(X*poshidprobsT – negdata*neghidprobsT)/bathSize
        timer.get().mark("W");
        return delta;
    }

    /**
     * neghidprobs := hidden unit probabilities given negdata (use Equation 3)
     *
     * @param negdata
     * @return
     */
    protected Matrix getNegHidProbs(Matrix negdata) {
        Matrix hidProbs = positivePhaseComputations.getHidProbs(negdata, sigma);
        timer.get().mark("neghidprobs");
        return hidProbs;
    }


    abstract public Matrix eval(Matrix matrix);

    public void persist(String s) {
        try (FileOutputStream fos = new FileOutputStream(s);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(layer);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public void load(String s) throws IOException, ClassNotFoundException {
        try (FileInputStream in = new FileInputStream(s)) {
            ObjectInputStream o = new ObjectInputStream(in);
            this.layer = (Layer) o.readObject();
        } catch (FileNotFoundException e) {
            Logger.getLogger(SRBM.class.getName()).info("File "+s+" not found. Starting from begining.");
        }
    }

    public Matrix W() {
        return layer.W;
    }

}
