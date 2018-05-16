package org.wit.snr.nn.srbm.math.collection;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.util.stream.Collectors.toList;

public class ProbabilisticVector {

    static final Random random = new Random();
    final List<Double> probabilities;
    final List<Boolean> sampledValues;
    final int size;

    public ProbabilisticVector(int size) {
        this.size = size;
        this.probabilities = new ArrayList<>(size);
        this.sampledValues = new ArrayList<>(size);
    }

    public List<Boolean> sampling() {
        if (probabilities.size() != size) {
            throw new IllegalStateException("Bad size of probabilities vector. " +
                    "This is fixed variant, size of probabilities list must be the same as this.size");
        }
        sampledValues.clear();
        List<Boolean> samplingResult = probabilities.stream()
                .map(d -> d > random.nextDouble())
                .collect(toList());
        sampledValues.addAll(samplingResult);
        return getSampledValues();
    }

    public List<Double> getProbabilities() {
        return probabilities;
    }

    public List<Boolean> getSampledValues() {
        if (sampledValues.isEmpty() || sampledValues.size() != size) {
            throw new IllegalStateException(
                    "Sampled values list size is different to ProbabilisticVector size. " +
                            "You must invoke sampling() method before. " +
                            "Or probabilities vector size is corupted.");
        } else {
            return sampledValues;
        }
    }
}
