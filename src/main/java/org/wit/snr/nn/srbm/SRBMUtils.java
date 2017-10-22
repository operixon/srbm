package org.wit.snr.nn.srbm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Created by kkoperkiewicz on 15.01.2017.
 */
public class SRBMUtils {

    public static Random rand = new Random();

    /**
     * Produce matrix collection witch random double values.
     *
     * @param rowsNumber
     * @param columnsNumber
     * @return
     */
    public static List<List<Double>> getRandomMatrix(final int rowsNumber, final int columnsNumber) {
        List<List<Double>> matrixCollection = new ArrayList<>();
        for (int rowIndex = 0; rowIndex < rowsNumber; rowIndex++) {
            List<Double> row = rand
                    .doubles(columnsNumber,0,1)
                    .boxed()
                    .collect(Collectors.toList());
            matrixCollection.add(row);
        }
        return matrixCollection;
    }
}
