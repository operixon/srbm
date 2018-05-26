package org.wit.snr.nn.srbm.trainingset;

import org.wit.snr.minst.MinstImageLoader;
import org.wit.snr.nn.srbm.TrainingSet;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;


public class TrainingSetMinst implements TrainingSet {

    final static Random rnd = new Random();
    final MinstImageLoader mil;
    final List<List<Double>> images;

    public TrainingSetMinst() throws IOException {
        String f = getClass().getClassLoader().getResource("t10k-images-idx3-ubyte").getPath();
        mil = new MinstImageLoader(f);
        images = Arrays.stream(mil.getImages())
                .map(image -> getNormalizedImageData(image))
                .collect(toList());
    }

    @Override
    /**
     *   training batch Xnumdims x batchSize (randomly sample batchSize patches from data w/o replacement)
     *   1. get all samples from training batch
     *   2. shufle order of images
     *   3. slice all images list to chunks
     *   4. convert single chunk to Matrix object
     *   5. Return list of chunk where chunk contains list of images
     */
    public List<Matrix> getTrainingBatch(int miniBatchSize) {
        final AtomicInteger counter = new AtomicInteger(0);
        Collections.shuffle(images);
        Collection<List<List<Double>>> minibatchesList = images.stream()
                .collect(Collectors.groupingBy(it -> counter.getAndIncrement() / miniBatchSize))
                .values();
        List<Matrix> collect = minibatchesList.stream().map(mibibatch -> new Matrix2D(mibibatch)).collect(Collectors.toList());
        return collect;
    }


    private List<Double> getNormalizedImageData(byte[][] image) {
        List<Double> convertedSample = new LinkedList<>();
        for (byte[] bytes : image) {
            for (byte aByte : bytes) {
                convertedSample.add(aByte == 0 ? 0.0 : 1.0);
            }
        }
        return convertedSample;
    }
}
