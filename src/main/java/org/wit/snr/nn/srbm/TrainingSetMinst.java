package org.wit.snr.nn.srbm;

import org.wit.snr.minst.MinstImageLoader;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix1D;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;


public class TrainingSetMinst implements TrainingSet {

    final MinstImageLoader mil;
    final static Random rnd = new Random();

    public TrainingSetMinst() throws IOException {
        String f = getClass().getClassLoader().getResource("t10k-images-idx3-ubyte").getPath();
        mil = new MinstImageLoader(f);
    }

    @Override
    /**
     *  training batch X numdims x batchSize
     */
    public Matrix getTrainingBatch(int batchSize) {
        List<Double> samples = new LinkedList<>();
        for (int i = 0; i < batchSize; i++) {
            int randomIndexOfImage = rnd.nextInt(mil.getNumberOfImages()) - 1;
            List<Double> sample = getNormalizedImageData(mil.getImage(randomIndexOfImage));
            samples.addAll(sample);
        }
        List<Double> samplesArray = new ArrayList<>(samples);
        int numdims = mil.getNumberOfRows() * mil.getNumberOfColumns();
        Matrix trainingBath = new Matrix1D(samplesArray, numdims, batchSize);
        return trainingBath;
    }


    private List<Double> getNormalizedImageData(byte[][] image) {
        List<Double> convertedSample = new LinkedList<>();
        for (byte[] bytes : image) {
            for (byte aByte : bytes) {
                convertedSample.add(aByte < 0 ? 0.0 : 1.0);
            }
        }
        return convertedSample;
    }
}
