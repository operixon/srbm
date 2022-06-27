package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.layer.Layer;
import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

public class SRBMMapReduceJSA extends SRBM {


    private SRBM prev;
    private SRBM next;

    public SRBMMapReduceJSA(RbmCfg cfg) throws IOException, InterruptedException {
        super(cfg);
    }

    @Override
    public Matrix eval(Matrix matrix) {
        if (prev != null) {
            return getHidProbs(prev.eval(matrix));
        } else {
            return getHidProbs(matrix);
        }
    }



    public SRBMMapReduceJSA(SRBM v1, RbmCfg cfg) throws IOException, InterruptedException {
        super(cfg);
        this.prev = v1;
        v1.setNext(this);

    }


    public void setPrev(SRBM prev) {
        this.prev = prev;
    }

    public SRBM setNext(SRBM next) {
        this.next = next;
        return next;
    }

    public void train(List<Matrix> x) {
        while (!isConverged()) {
            epoch(x);
        }

    }

    private void epoch(List<Matrix> x) {
        Collections.shuffle(x);
        getMapReduceResult(x);
        updateSigma();
        miniBatchIndex.set(0);
        currentEpoch.incrementAndGet();
    }

    private void getMapReduceResult(List<Matrix> x) {
        if (prev != null) {
            x.parallelStream()
               //  .limit(5)
                 .map(prev::eval)
                 .map(this::trainMiniBatch)
                 .filter(Optional::isPresent)
                 .map(Optional::get)
                 .peek(this::updateLayerData)
                 .count();
        } else {
            x.parallelStream()
              //   .limit(5)
                 .map(this::trainMiniBatch)
                 .filter(Optional::isPresent)
                 .map(Optional::get)
                 .peek(this::updateLayerData)
                 .count();
        }
        //.reduce(
        //      Optional.empty(),
        //    (sum, nextValue) -> sum.map(s -> s.apply(nextValue))
        //          .orElse(nextValue)
        //);
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


    private Optional<MiniBatchTrainingResult> trainMiniBatch(Matrix X) {
        final int batchIndex = miniBatchIndex.getAndIncrement();
        // timer.set(new Timer());
        timer.get().start();

        Matrix poshidprobs = getHidProbs(X);
        Matrix poshidstates = getHidStates(poshidprobs);
        Matrix negdata = getNegData(poshidprobs);
        Matrix neghidprobs = getNegHidProbs(negdata);


        Matrix Wdelta = updateWeights(X, poshidprobs, negdata, neghidprobs);
        Matrix vBiasDelta = updateVBias(X, negdata);
        updateError(X, negdata);
        Matrix hBiasDelta = getHBiasDelta(X);

        System.out.printf("E %s/%s | %s | %s %n",
                          batchIndex * cfg.batchSize(),
                          currentEpoch,
                          layer.error,
                          timer.get().toString());

        datavis datavis = new datavis(
                X,
                batchIndex,
                layer,
                poshidprobs,
                poshidstates,
                negdata,
                neghidprobs,
                Wdelta,
                vBiasDelta,
                hBiasDelta
        );
        draw(datavis);
        timer.remove();
        return Optional.of(new MiniBatchTrainingResult(Wdelta, vBiasDelta, hBiasDelta));
    }

    protected boolean isConverged() {
        return currentEpoch.get() > cfg.numberOfEpochs() || layer.error < cfg.acceptedError();
    }

    public Layer getLayer() {
        return this.layer;
    }
}
