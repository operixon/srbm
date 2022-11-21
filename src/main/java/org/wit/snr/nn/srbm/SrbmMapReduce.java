package org.wit.snr.nn.srbm;

import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.rdd.RDD;
import org.wit.snr.nn.srbm.layer.Model;

import java.io.IOException;
import java.util.*;

public class SrbmMapReduce extends AbstractSrbm {

    public SrbmMapReduce(RbmCfg cfg) throws IOException, InterruptedException {
        super(cfg);
    }


    public void fit(RDD<Vector> x) {
        while (!isConverged()) {
            var splitedData = splitToBatches(x);
            epoch(splitedData);
        }
    }

    private RDD<Vector[]> splitToBatches(RDD<Vector> x) {
        trainingSetSize = x.count();
        int numberOffPArtitions = (int) (trainingSetSize / cfg.batchSize());
        return x.toJavaRDD().repartition(numberOffPArtitions).rdd().glom();
    }

    private void epoch(RDD<Vector[]> x) {
        mapReduceWorkflow(x);
        updateSigma();
        miniBatchIndex.set(0);
        currentEpoch.incrementAndGet();

    }

    private void mapReduceWorkflow(RDD<Vector[]> x) {

        x.toJavaRDD()
         .map(this::trainMiniBatch)
         .filter(Optional::isPresent)
         .map(Optional::get)
         .foreach(this::updateLayerData);
    }

    private void updateSigma() {
        if (sigma > 0.05)
            sigma = sigma * cfg.sigmaDecay();
    }

    private void updateLayerData(MiniBatchTrainingResult delta) {
        synchronized (layer) {
            layer.W = layer.W.matrixAdd(delta.getW());
            layer.hbias = layer.hbias.matrixAdd(delta.getHbias());
            layer.vbias = layer.vbias.matrixAdd(delta.getVbias());
        }
    }


    private Optional<MiniBatchTrainingResult> trainMiniBatch(Vector[] X) {
        final int batchIndex = miniBatchIndex.getAndIncrement();
        // timer.set(new Timer());

        var poshidprobs = getHidProbs(X);
        var poshidstates = getHidStates(poshidprobs);
        var negdata = getNegData(poshidstates);
        var neghidprobs = getNegHidProbs(negdata);


        Matrix Wdelta = updateWeights(X, poshidprobs, negdata, neghidprobs);
        Matrix vBiasDelta = updateVBias(X, negdata);
        updateError(X, negdata);
        Matrix hBiasDelta = getHBiasDelta(X);

        System.out.printf("%s | sample : %s/%s | epoch : %s/%s | error: %.5f | Timer => %s %n",
                          cfg.name(),
                          batchIndex * cfg.batchSize(),
                          trainSetSize,
                          currentEpoch,
                          cfg.numberOfEpochs(),
                          layer.error,
                          timer.get().toString());


        timer.remove();
        synchronized (epochHandlersList) {
            epochHandlersList.forEach(h -> h.accept(this.layer));
        }
        return Optional.of(new MiniBatchTrainingResult(Wdelta, vBiasDelta, hBiasDelta));
    }

    protected boolean isConverged() {
        return currentEpoch.get() > cfg.numberOfEpochs() || layer.error < cfg.acceptedError();
    }

    public Model getLayer() {
        return this.layer;
    }

}
