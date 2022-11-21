package org.wit.snr.nn.srbm;

import org.apache.spark.mllib.linalg.Matrix;
import org.wit.snr.nn.srbm.layer.Layer;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class SrbmMapReduce extends SrbmLayer {


    private SrbmLayer prev;
    private SrbmLayer next;

    public SrbmMapReduce(RbmCfg cfg) throws IOException, InterruptedException {
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


    public SrbmMapReduce(SrbmLayer v1, RbmCfg cfg) throws IOException, InterruptedException {
        super(cfg);
        this.prev = v1;
        v1.setNext(this);

    }


    public void setPrev(SrbmLayer prev) {
        this.prev = prev;
    }

    public SrbmLayer setNext(SrbmLayer next) {
        this.next = next;
        return next;
    }

    public void train(List<List<Double>> x) {
        trainSetSize = x.size();
        while (!isConverged()) {
            epoch(splitToBatches(x));
        }

    }

    private List<Matrix> splitToBatches(List<List<Double>> x) {
        final AtomicInteger counter = new AtomicInteger(0);
        Collections.shuffle(x);
        Collection<List<List<Double>>> minibatchesList = x
                .stream()
                .collect(Collectors.groupingBy(it -> counter.getAndIncrement() / cfg.batchSize()))
                .values();
        return minibatchesList.stream().map(Matrix2D::new).collect(Collectors.toList());
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

    public Layer getLayer() {
        return this.layer;
    }

}
