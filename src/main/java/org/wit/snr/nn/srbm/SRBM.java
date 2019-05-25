/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.layer.*;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;
import org.wit.snr.nn.srbm.math.function.GausianDensityFunction;
import org.wit.snr.nn.srbm.math.function.SigmoidFunction;
import org.wit.snr.nn.srbm.monitoring.Timer;
import org.wit.snr.nn.srbm.trainingset.TrainingSetMinst;
import org.wit.snr.nn.srbm.visualization.MatrixRenderer;
import org.wit.snr.nn.srbm.visualization.MatrixRendererHiddenUnits;
import org.wit.snr.nn.srbm.visualization.MatrixRendererIF;
import org.wit.snr.nn.srbm.visualization.MatrixRendererSample;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

/**
 * @author koperix
 */
public abstract class SRBM {

    final Configuration cfg;


    final Layer layer;
    final TrainingSet trainingSet;
    final PositivePhaseComputations positivePhaseComputations;
    final NegativePhaseComputations negativePhaseComputations;
    final HiddenBiasAdaptation hiddenBiasAdaptation;

    final ThreadLocal<Timer> timer = new ThreadLocal();
    final AtomicInteger currentEpoch = new AtomicInteger(0);
    final AtomicInteger miniBatchIndex = new AtomicInteger(0);

    JFrame frame; // TODO : refactor
    Canvas canvas; // TODO : refacotr

    final String sessionId = "srbm-" + System.currentTimeMillis();

    protected double sigma;
    protected SRBM previousLayer;

    public abstract void train();

    public SRBM(Configuration cfg) throws IOException, InterruptedException {
        this.cfg = cfg;
        this.layer = new Layer(cfg.numdims(), cfg.numhid());
        trainingSet = new TrainingSetMinst();
        Equation3 equation3 = new Equation3(cfg, layer, new SigmoidFunction());
        positivePhaseComputations = new PositivePhaseComputations(equation3, cfg);
        hiddenBiasAdaptation = new HiddenBiasAdaptation(equation3);
        negativePhaseComputations = new NegativePhaseComputations(
                new Equation2(cfg, layer, new SigmoidFunction()),
                cfg
        );
        initCanvas();
        sigma = cfg.sigmaInit();
    }

    private void initCanvas() {
        frame = new JFrame("sRBM");
        frame.setSize(1500, 700);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setResizable(true);
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

    void draw(datavis datavis) {
        displayVisualizationOnScreen(datavis);
        if (cfg.saveVisualization()) saveVisualizationToFile(datavis);
    }

    private void displayVisualizationOnScreen(datavis d) {
        synchronized (canvas) {
            BufferStrategy bufferStrategy = canvas.getBufferStrategy();
            Graphics graphics = bufferStrategy.getDrawGraphics();
            renderVisualizationOnGraphicsComponent(d, graphics);
            bufferStrategy.show();
            graphics.dispose();
        }
    }

    private void renderVisualizationOnGraphicsComponent(datavis d, Graphics graphics) {
        graphics.clearRect(0, 0, 1700, 1200);
        int visSize = (int) Math.sqrt(cfg.numdims());
        int hidSize = (int) Math.sqrt(cfg.numhid());
        MatrixRendererIF[] rlist = {
                // Wagi
                new MatrixRenderer(0, 10, visSize, d.layer.W, graphics),
                // biasy
                new MatrixRendererHiddenUnits(0, 1100, d.vBiasDelta.reshape(visSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(100, 1100, d.layer.vbias.reshape(visSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(0, 1000, d.hBiasDelta.reshape(hidSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(100, 1000, d.layer.hbias.reshape(hidSize).transpose(), graphics),
                // data flow
                new MatrixRendererHiddenUnits(200, 610, d.X.reshape(visSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(300, 610, d.poshidprobs.reshape(hidSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(400, 610, d.poshidstates.reshape(hidSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(500, 610, d.negdata.reshape(visSize).transpose(), graphics),
                new MatrixRendererHiddenUnits(600, 610, d.neghidprobs.reshape(hidSize).transpose(), graphics),

        };

        for (MatrixRendererIF matrixRendererIF : rlist) {
            matrixRendererIF.render();
        }

    }

    public void saveVisualizationToFile(datavis d) {
        try {
            BufferedImage image = new BufferedImage(canvas.getWidth(), canvas.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D graphics = image.createGraphics();
            renderVisualizationOnGraphicsComponent(d, graphics);
            graphics.dispose();
            String pathname = cfg.visualizationOutDirectory()
                    + File.separatorChar
                    + sessionId
                    + "-" + currentEpoch.get()
                    + "-" + d.bathIdx
                    + ".jpg";
            ImageIO.write(image, "JPEG", new File(pathname));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    protected Matrix getNegData(Matrix poshidstates) {
        Matrix negData = negativePhaseComputations.getNegData(poshidstates, sigma);
        timer.get().mark("negdata");
        return negData;
    }

    protected List<Matrix> getTrainingBatch() {
        List<Matrix> trainingBatch = trainingSet.getTrainingBatch(cfg.batchSize());
        return trainingBatch;
    }

    protected List<Matrix> getTrainingBatch2() {
        List<Matrix> trainingSet = new LinkedList<>();

        for (int t = 0; t < 100; t++) {
            List<List<Double>> batch = new LinkedList<>();
            boolean s = true;
            for (int imageIndex = 0; imageIndex < cfg.batchSize(); imageIndex++) {
                List<Double> image = new LinkedList<>();
                for (int i = 0; i < 28 / 4; i++) {
                    List<Double> w = Matrix2D.createFilledMatrix(2, 28, 0).getDataAsList();
                    List<Double> b = Matrix2D.createFilledMatrix(2, 28, 1).getDataAsList();
                    image.addAll(w);
                    image.addAll(b);
                }
                if (image.size() != 784) {
                    throw new IllegalStateException();
                }
                if (s) {
                    image = Matrix2D.createColumnVector(image).reshape(28).transpose().getDataAsList();
                }
                batch.add(image);

                s = !s;
            }
            Matrix2D batchMatrix = new Matrix2D(batch);
            if (batchMatrix.getColumnsNumber() != cfg.batchSize()) {
                throw new IllegalStateException();
            }
            if (batchMatrix.getRowsNumber() != cfg.numdims()) {
                throw new IllegalStateException();
            }
            trainingSet.add(batchMatrix);
        }
        return trainingSet;
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

    public Layer getLayer() {
        return layer;
    }

    public void connectPreviousLayer(SRBM prevLayer) {
        this.previousLayer = prevLayer;
    }

    protected Matrix propagate(Matrix x) {
        return positivePhaseComputations.getHidProbs(x, sigma);
    }

    protected Matrix backPropagate(Matrix x)
    {
        Matrix negData = negativePhaseComputations.getNegData(x, sigma);


        return negData;
    }
}