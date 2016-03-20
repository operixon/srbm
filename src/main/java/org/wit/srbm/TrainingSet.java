/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.srbm;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author koperix
 */
class TrainingSet {

    final DoubleMatrix1D[] data;
    final int numdims;
    final int setSize;

    private static final Map<Integer, DoubleMatrix1D> types = new HashMap<>();

    static {
        types.put(1, DoubleFactory1D.dense.make(
                new double[]{
                    0, 0, 1, 0, 0,//
                    0, 1, 1, 0, 0,//
                    0, 0, 1, 0, 0,//
                    0, 1, 1, 1, 0//
                }
        ));
        types.put(0, DoubleFactory1D.dense.make(
                new double[]{
                    0, 1, 1, 1, 0,//
                    0, 1, 0, 1, 0,//
                    0, 1, 0, 1, 0,//
                    0, 1, 1, 1, 0//
                }
        ));
        types.put(7, DoubleFactory1D.dense.make(
                new double[]{
                    0, 1, 1, 1, 1,//
                    0, 0, 0, 1, 0,//
                    0, 0, 1, 0, 0,//
                    0, 1, 0, 0, 0//
                }
        ));
        types.put(4, DoubleFactory1D.dense.make(
                new double[]{
                    0, 1, 0, 1, 0,//
                    0, 1, 1, 1, 0,//
                    0, 0, 0, 1, 0,//
                    0, 0, 0, 1, 0//
                }
        ));

    }

    public TrainingSet(int numdims, int setSize) {
        data = genData();
        this.numdims = numdims;
        this.setSize = setSize;
    }

    public void init() {

    }

    DoubleMatrix1D[] getBatchOffRandomlySamples(int batchSize) {

        // Junkcode
        DoubleMatrix1D [] s = new DoubleMatrix1D[batchSize];
        Arrays.fill(s, types.get(0));
        return s;
    
    }

    private DoubleMatrix1D[] genData() {
        return null;
    }

}
