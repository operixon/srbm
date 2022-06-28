package org.wit.snr.nn.srbm.math;

import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.util.List;
import java.util.Random;

import static java.util.stream.Collectors.toList;

public class RandomSampler {

    final private static Random random = new Random();

    public List<Double> sample(List<Double> probs) {
        return probs.stream()
                    .map(cel -> cel > random.nextDouble() ? 1.0 : 0.0)
                    .collect(toList());
    }
}
