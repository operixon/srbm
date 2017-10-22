package org.wit.snr.nn.srbm;

import org.testng.annotations.Test;

import java.util.List;

import static org.testng.Assert.*;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class SRBMUtilsTest {

    @Test
    public void random_matrix_size_test(){
        List<List<Double>> randomMatrix = SRBMUtils.getRandomMatrix(0, 0);

    }

}