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
                            .map(x -> rand.nextGaussian() * 0.01)
                            .collect(Collectors.toList())
            );
        }
        return columnsList;
    }
}
