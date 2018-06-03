package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.monitoring.Timer;

import java.io.IOException;

public class SRBMMapReduceJSA extends SRBM {


    public SRBMMapReduceJSA() throws IOException, InterruptedException {
        super();
    }

    private static MiniBatchTrainingResult applyDeltas(MiniBatchTrainingResult p1, MiniBatchTrainingResult p2) {
        return new MiniBatchTrainingResult(
                p1.getW().matrixAdd(p2.getW()),
                p1.getVbias().matrixAdd(p2.getVbias()),
                p1.getHbias().matrixAdd(p2.getHbias())
        );
    }

    public void train() {
        while (isConverged()) {
            MiniBatchTrainingResult epochTrainnigResult = getTrainingBatch()
                    .parallelStream()
                    .map(x -> trainMiniBatch(x))
                    .reduce(new MiniBatchTrainingResult(layer.W, layer.vbias, layer.hbias),
                            SRBMMapReduceJSA::applyDeltas);
            updateLayerData(epochTrainnigResult);
            miniBatchIndex.set(0);
            if (cfg.sigma > 0.05)
                cfg.sigma = cfg.sigma * 0.99;

        }//#while end

        // embarrassing quality solution to prevent closing jframe after end of learning
        try {
            Thread.sleep(1000 * 60 * 60 * 60 * 24);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }//#train_rbm

    private void updateLayerData(MiniBatchTrainingResult reduce) {
        layer.W = reduce.getW();
        layer.hbias = reduce.getHbias();
        layer.vbias = reduce.getVbias();
        currentEpoch.incrementAndGet();
    }


    private MiniBatchTrainingResult trainMiniBatch(Matrix X) {
        final int batchIndex = miniBatchIndex.getAndIncrement();
        timer.set(new Timer());
        timer.get().start();
        Matrix poshidprobs = getHidProbs(X);
        Matrix poshidstates = getHidStates(poshidprobs);
        Matrix negdata = getNegData(poshidstates);
        Matrix neghidprobs = getNegHidProbs(negdata);
        Matrix Wdelta = updateWeights(X, poshidprobs, negdata, neghidprobs);
        Matrix vBiasDelta = updateVBias(X, negdata);
        updateError(X, negdata);
        Matrix hBiasDelta = updateHBias(X);

        System.out.printf("E %s/%s | %s | %s %n", batchIndex * cfg.batchSize, currentEpoch, layer.error, timer.get().toString());
        draw(batchIndex, layer.W, X, negdata, layer.hbias.reshape(50), layer.vbias);
        // timer.get().reset();
        timer.remove();
        return new MiniBatchTrainingResult(Wdelta, vBiasDelta, hBiasDelta);
    }

    protected boolean isConverged() {
        return currentEpoch.get() < cfg.numberOfEpochs;
    }
}
