package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.monitoring.Timer;

import java.io.IOException;
import java.util.Optional;

public class SRBMMapReduceJSA extends SRBM
{


    public SRBMMapReduceJSA() throws IOException, InterruptedException
    {
        super();
    }

    public void train()
    {
        while (isConverged())
        {
            epoch();
        }
    }

    private void epoch()
    {
        getMapReduceResult().ifPresent(cumulatedDelta ->
                                               updateLayerData(cumulatedDelta));
        updateSigma();
        miniBatchIndex.set(0);
        currentEpoch.incrementAndGet();
    }

    private Optional<MiniBatchTrainingResult> getMapReduceResult()
    {
        return getTrainingBatch()
                .parallelStream()
                .map( samples -> trainMiniBatch(samples))
                .reduce(
                        Optional.empty(),
                        (sum, nextValue) -> sum.map(s -> s.apply(nextValue))
                                               .orElse(nextValue)
                       );
    }

    private void updateSigma()
    {
        if (cfg.sigma > 0.05)
            cfg.sigma = cfg.sigma * 0.99;
    }

    private void updateLayerData(MiniBatchTrainingResult delta)
    {
        layer.W = layer.W.matrixAdd(delta.getW());
        layer.hbias = layer.hbias.matrixAdd(delta.getHbias());
        layer.vbias = layer.vbias.matrixAdd(delta.getVbias());
    }


    private Optional<MiniBatchTrainingResult> trainMiniBatch(Matrix X)
    {
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
        timer.remove();
        return Optional.of(new MiniBatchTrainingResult(Wdelta, vBiasDelta, hBiasDelta));
    }

    protected boolean isConverged()
    {
        return currentEpoch.get() < cfg.numberOfEpochs;
    }

}
