package org.wit.snr.nn.srbm.math;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by kkoperkiewicz on 15.01.2017.
 */
public class MathUtils {

    public static Random rand = new Random();

    public static List<Double> getRandomList(final int length) {
        List<Double> collect =
                Stream
                        .iterate(0, i -> i++)
                        .limit(length)
                        .map(x -> (rand.nextGaussian() / 10) - 0.005)
                        .collect(Collectors.toList());
        return new ArrayList<Double>(collect);
    }

    /**
     * Produce matrix collection witch random double values.
     *
     * @param rowsNumber
     * @param columnsNumber
     * @return
     */
    public static List<List<Double>> getRandomMatrix(final int rowsNumber, final int columnsNumber) {
        List<List<Double>> columnsList = new ArrayList<>();
        for (int columnIdx = 0; columnIdx < columnsNumber; columnIdx++) {
            columnsList.add(
                    Stream.iterate(0, i -> i = i + 1)
                            .limit(rowsNumber)
                            .map(x -> rand.nextGaussian())
                            .collect(Collectors.toList())
            );
        }
        return columnsList;
    }
}
