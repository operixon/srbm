package org.wit.snr.nn.srbm;

import java.util.List;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class Vector {

    private double[] doubleData;
    private List<Boolean> booleanListData;

    public Vector(double[] poshidprobs) {
        doubleData = poshidprobs;
    }

    public Vector(List<Boolean> negdata) {

        booleanListData = negdata;
    }

    public Matrix multiplyByTransposed(Vector poshidprobsVector) {

    }
}
