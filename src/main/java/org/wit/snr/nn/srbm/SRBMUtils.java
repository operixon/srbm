package org.wit.snr.nn.srbm;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by kkoperkiewicz on 15.01.2017.
 */
public class SRBMUtils {

    public static Random rand = new Random();

    public static double[][] getRandomMatrix(int w, int h) {
        double [][] m = new double[w][];
        Arrays.stream(m).forEach(col -> col = rand.doubles(h,0,1).toArray());
        return m;
    }
}
