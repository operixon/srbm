package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.monitoring.Timer;

import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class SRBMMapReduceJSA extends SRBM {


    private List<Matrix> batch;
    private SRBM prev;
    private SRBM next;

    public SRBMMapReduceJSA(Configuration cfg) throws IOException, InterruptedException {
        super(cfg);
        batch = getTrainingBatch();
    }

    @Override
    public Matrix eval(Matrix matrix) {
        if (prev != null) {
            return getHidProbs(prev.eval(matrix));
        } else {
            return getHidProbs(matrix);
        }
    }

    public SRBMMapReduceJSA(SRBM v1,Configuration cfg) throws IOException, InterruptedException {
        super(cfg);
        batch = getTrainingBatch();
        this.prev = v1;
        v1.setNext(this);

    }


    public void setPrev(SRBM prev) {
        this.prev = prev;
    }

    public void setNext(SRBM next) {
        this.next = next;
    }

    public void train() {
        while (!isConverged()) {
            epoch();
        }
        if (next != null) {
            next.train();
        }
    }

    private void epoch() {
        getMapReduceResult();
        updateSigma();
        miniBatchIndex.set(0);
        currentEpoch.incrementAndGet();
        batch = getTrainingBatch();
    }

    private void getMapReduceResult() {
        if (prev != null) {
            batch.parallelStream()
                    .limit(1)
                    .map(prev::eval)
                    .map(this::trainMiniBatch)
                    .filter(Optional::isPresent)
                    .map(Optional::get)
                    .peek(this::updateLayerData)
                    .count();
        } else {
            batch.parallelStream()
                    .limit(1)
                    .map(samples -> trainMiniBatch(samples))
                    .peek(tbr -> updateLayerData(tbr.get()))
                    .collect(Collectors.counting());
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

}
