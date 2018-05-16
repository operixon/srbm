package org.wit.snr.nn.srbm;

import org.testng.annotations.Test;
import org.wit.snr.nn.srbm.math.MathUtils;

import java.util.List;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class SRBMUtilsTest {

    @Test
    public void random_matrix_size_test(){
        List<List<Double>> randomMatrix = MathUtils.getRandomMatrix(0, 0);

    }

}