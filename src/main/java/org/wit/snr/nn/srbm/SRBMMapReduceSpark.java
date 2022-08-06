package org.wit.snr.nn.srbm;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.wit.snr.nn.srbm.layer.Layer;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class SRBMMapReduceSpark extends SRBM {


    private SRBM prev;
    private SRBM next;
    private List<Consumer<Layer>> epochHandlersList = new LinkedList<>();
    private int trainSetSize = 0;
    private JavaSparkContext sc;


    public SRBMMapReduceSpark(RbmCfg cfg) throws IOException, InterruptedException {
        super(cfg);
        //
        // The "modern" way to initialize Spark is to create a SparkSession
        // although they really come from the world of Spark SQL, and Dataset
        // and DataFrame.
        //
        SparkSession spark = SparkSession
                .builder()
                .appName("RDD-Basic")
                .master("local[8]")
                .getOrCreate();
        //
        // Operating on a raw RDD actually requires access to the more low
        // level SparkContext -- get the special Java version for convenience
        //
        sc = new JavaSparkContext(spark.sparkContext());
    }

    @Override
    public Matrix eval(Matrix matrix) {
        if (prev != null) {
            return getHidProbs(prev.eval(matrix));
        } else {
            return getHidProbs(matrix);
        }
    }




    public SRBMMapReduceSpark(SRBM v1, RbmCfg cfg) throws IOException, InterruptedException {
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
            sc.parallelize(x)
              //  .limit(5)
              .map(prev::eval)
              .map(this::trainMiniBatch)
              .filter(Optional::isPresent)
              .map(Optional::get)
              .foreach(this::updateLayerData);
        } else {
            sc.parallelize(x)
              .map(this::trainMiniBatch)
              .filter(Optional::isPresent)
              .map(Optional::get)
              .foreach(this::updateLayerData);
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
        Matrix negdata = getNegData(poshidstates);
        Matrix neghidprobs = getNegHidProbs(negdata);


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

    public void addHook(Consumer<Layer> c) {
        this.epochHandlersList.add(c);
    }
}
