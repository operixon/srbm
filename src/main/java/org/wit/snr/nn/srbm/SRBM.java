/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.ActivationFunction;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;
import org.wit.snr.nn.srbm.math.function.GausianDensityFunction;
import org.wit.snr.nn.srbm.math.function.SigmoidFunction;
import org.wit.snr.nn.srbm.monitoring.Timer;
import org.wit.snr.nn.srbm.trainingset.TrainingSetMinst;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.io.IOException;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

/**
 * @author koperix
 */
public class SRBM {

    final Configuration cfg = new Configuration();
    final Layer layer;
    final ActivationFunction gausianDensityFunction;
    final TrainingSet trainingSet;
    final HiddenLayerComputations hiddenLayerComputations;
    final HiddenBiasAdaptation hiddenBiasAdaptation;
    final Timer timer;
    int currentEpoch;
    JFrame frame;
    Canvas canvas;
    Graphics graphics;

    public SRBM() throws IOException {
        this.layer = new Layer(cfg.numdims, cfg.numhid);
        gausianDensityFunction = new GausianDensityFunction(cfg.mi);
        trainingSet = new TrainingSetMinst();
        Equation3 equation3 = new Equation3(cfg, layer, new SigmoidFunction());
        hiddenLayerComputations = new HiddenLayerComputations(equation3, cfg);
        hiddenBiasAdaptation = new HiddenBiasAdaptation(equation3);
        timer = new Timer();
        initCanvas();
    }

    private void initCanvas() {
        frame = new JFrame("sRBM");

        frame.setSize(1500, 700);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setResizable(false);
        frame.setVisible(true);

        //Creating the canvas.
        canvas = new Canvas();

        canvas.setSize(700, 500);
        canvas.setBackground(Color.BLACK);
        canvas.setVisible(true);
        canvas.setFocusable(false);


        //Putting it all together.
        frame.add(canvas);

        canvas.createBufferStrategy(3);


    }

    void draw(Matrix W, Matrix X, Matrix negM, Matrix neghidprobs, Matrix vbias) {

        BufferStrategy bufferStrategy;
        bufferStrategy = canvas.getBufferStrategy();
        graphics = bufferStrategy.getDrawGraphics();
        graphics.clearRect(0, 0, 1700, 1200);


        MatrixRenderer Wr = new MatrixRenderer(0, 10, W, graphics);
        MatrixRenderer neg = new MatrixRenderer(680, 10, negM, graphics);
        MatrixRendererHiddenUnits neghidprobsDraw = new MatrixRendererHiddenUnits(600, 5, neghidprobs, graphics);
        MatrixRendererSample Xprint = new MatrixRendererSample(680, 350, X, graphics, Color.WHITE);
        MatrixRenderer vbiasDraw = new MatrixRenderer(650, 200, vbias, graphics);
        Wr.render();
        Xprint.render();
        neg.render();
        neghidprobsDraw.render();
        vbiasDraw.render();


        bufferStrategy.show();
        graphics.dispose();
    }


    public void train() {
        currentEpoch = 0;
        while (isConverged()) {

            for (Matrix X : getTrainingBatch()) {
                timer.start();
                Matrix poshidprobs = getHidProbs(X);
                Matrix poshidstates = getHidStates(poshidprobs);
                Matrix negdata = getNegData(poshidstates);
                Matrix neghidprobs = getNegHidProbs(negdata.gibsSampling());
                updateWeights(X, poshidprobs, negdata.gibsSampling(), neghidprobs);
                updateVBias(X, negdata);
                updateError(X, negdata);
                updateHBias(X);
                System.out.printf("E %s | %s | %s %n", currentEpoch, layer.error, timer.toString());
                draw(layer.W, X, negdata, layer.hbias.reshape(80), layer.vbias);
                timer.reset();
            }
            currentEpoch++;
            // Zgodnie z algorytmem
            // update hbias (use Equation 6)
            // powinno być w tym miejscu ale wtedy nie mam dostępu do
            // paczki trenującej

            if (cfg.sigma > 0.05) cfg.sigma = cfg.sigma * 0.99;

        }//#while end

        // Stiupid hack to prevent closing jframe after end of learning
        try {
            Thread.sleep(1000 * 60 * 60 * 60 * 24);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }//#train_rbm

    private List<Matrix> getTrainingBatch() {
        List<Matrix> trainingBatch = trainingSet.getTrainingBatch(cfg.batchSize);
        timer.mark("X");
        return trainingBatch;
    }

    private Matrix getHidStates(Matrix poshidprobs) {
        Matrix hidStates = hiddenLayerComputations.getHidStates(poshidprobs);
        timer.mark("hidstates");
        return hidStates;
    }

    private Matrix getHidProbs(Matrix X) {
        Matrix hidProbs = hiddenLayerComputations.getHidProbs(X);
        timer.mark("hidprobs");
        return hidProbs;
    }

    private boolean isConverged() {
        return currentEpoch < cfg.numberOfEpochs;
    }

    private void updateHBias(Matrix X) {
        List<Double> updatedBiasData = Stream
                .iterate(0, j -> j = j + 1)
                .limit(cfg.numhid)
                .map(j ->
                        hiddenBiasAdaptation.getHiddenBiasUnit(
                                layer.vbias.get(j, 0),
                                cfg.alpha,
                                cfg.batchSize,
                                j,
                                cfg.sparsneseFactor,
                                X))
                .collect(toList());
        layer.hbias = Matrix2D.createColumnVector(updatedBiasData);
        timer.mark("hbias");
    }


    /**
     * error := SquaredDiff(X,negdata)
     *
     * @param X
     * @param negdata
     */
    private void updateError(Matrix X, Matrix negdata) {

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
        layer.error = error / (cfg.batchSize * cfg.numdims);
        timer.mark("error");
    }

    /**
     * vbias := vbias + α(rowsum(X) – rowsum(negdata))/batchSize
     *
     * @param X
     * @param negdata
     */
    private void updateVBias(Matrix X, Matrix negdata) {
        Matrix rowsum_X = X.rowsum();
        Matrix rowsum_negdata = negdata.rowsum();
        Matrix biasDelta = rowsum_X.subtract(rowsum_negdata).scalarMultiply(cfg.alpha / (double) cfg.batchSize);
        layer.vbias = layer.vbias.matrixAdd(biasDelta);
        timer.mark("vbias");
    }

    /**
     * W := W + α(X*poshidprobsT – negdata*neghidprobsT)/batchSize
     *
     * @param X           visible layer samples batch
     * @param poshidprobs positive phase hidden layer probabilities batch
     * @param negdata     visible layer batch data from negative phase
     * @param neghidprobs hidden layer probabilities for negative phase
     */
    private void updateWeights(Matrix X, Matrix poshidprobs, Matrix negdata, Matrix neghidprobs) {
        // X*poshidprobsT
        Matrix x_poshidprobsT = X.multiplication(poshidprobs.transpose());
        // negdata*neghidprobsT
        Matrix negdata_neghidprobsT = negdata.multiplication(neghidprobs.transpose());
        // (X*poshidprobsT – negdata*neghidprobsT)
        Matrix x_poshidprobsT_negdata_neghidprobsT = x_poshidprobsT.subtract(negdata_neghidprobsT);
        // a*(X*poshidprobsT – negdata*neghidprobsT)/bathSize
        Matrix delta = x_poshidprobsT_negdata_neghidprobsT.scalarDivide((double) cfg.batchSize).scalarMultiply(cfg.alpha);
        layer.W = layer.W.matrixAdd(delta);
        timer.mark("W");
    }

    /**
     * neghidprobs := hidden unit probabilities given negdata (use Equation 3)
     *
     * @param negdata
     * @return
     */
    private Matrix getNegHidProbs(Matrix negdata) {
        Matrix hidProbs = hiddenLayerComputations.getHidProbs(negdata);
        timer.mark("neghidprobs");
        return hidProbs;
    }

    /**
     * <pre>
     *
     * negdata := reconstruction of visible values given poshidstates (use Equation 2)
     * Iterate by N hidden samples
     *      For N-th hidden sample
     *              Iterate by all visible layer units
     *                  for i visible unit execute equation 2
     *
     * </pre>
     *
     * @param poshidstates matrix of hidden units states from positive phase
     * @return matrix of visible layers of reconstructed data without gibs sampling ( it is beater to use probabilities than boolean samples)
     */
    private Matrix getNegData(Matrix poshidstates) {
        List<List<Double>> visibleUnitsProbs = poshidstates
                .getMatrixAsCollection()
                .stream()
                .map( // For each hidden layer reproduce visual layer unit by unit using equation nr 2
                        hiddenLayerStates -> IntStream
                                .range(0, cfg.numdims)
                                .mapToDouble(visibleUnitIndex -> equation2(visibleUnitIndex, hiddenLayerStates))
                                .boxed()
                                .collect(toList())
                )
                .collect(toList());
        Matrix hp = new Matrix2D(visibleUnitsProbs);
        if (hp.getRowsNumber() != cfg.numdims || hp.getColumnsNumber() != cfg.batchSize) {
            throw new IllegalStateException(String.format("Matrix incorrect size. Expected size %dx%d. Actual %s", cfg.batchSize, cfg.numhid, hp));
        }
        timer.mark("negdata");
        return hp;
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
    private Double equation2(final int i, List<Double> hiddenUnitStates) {
        double x = cfg.lambda * layer.getActivationSignalForVisibleUnit(hiddenUnitStates, i);
        return gausianDensityFunction.evaluate(x, cfg.sigma);

    }


}
