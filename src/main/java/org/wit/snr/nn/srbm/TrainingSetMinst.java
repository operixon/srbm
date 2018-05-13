package org.wit.snr.nn.srbm;

import org.wit.snr.minst.MinstImageLoader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class TrainingSetMinst implements TrainingSet<List<Boolean>> {

    final MinstImageLoader mil;

    public TrainingSetMinst() throws IOException {
        String f = getClass().getClassLoader().getResource("t10k-images-idx3-ubyte").getPath();
        mil = new MinstImageLoader(f);
    }

    @Override
    public List<List<Boolean>> getTrainingBatch(int batchSize) {
        List<List<Boolean>> samples = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            List<Boolean> sample = convert(mil.getImage(i));
            samples.add(sample);
        }
        return samples;

    }

    private List<Boolean> convert(byte[][] image) {
        List<Boolean> convertedSample = new ArrayList<>();
        for (byte[] bytes : image) {
            for (byte aByte : bytes) {
                convertedSample.add(aByte < 0 ? Boolean.FALSE : Boolean.TRUE);
            }
        }
        return convertedSample;
    }
}
