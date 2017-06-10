/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;

import java.util.*;

/**
 * @author koperix
 */
class TrainingSetMock_DIM20_BOOLEAN implements TrainingSet<List<Boolean>> {

    private static final Map<Integer, List<Boolean>> types = new HashMap<>();

    static {
        types.put(1, new ArrayList<>(Arrays.asList(
                FALSE, FALSE, TRUE, FALSE, FALSE,//
                FALSE, TRUE, TRUE, FALSE, FALSE,//
                FALSE, FALSE, TRUE, FALSE, FALSE,//
                FALSE, TRUE, TRUE, TRUE, FALSE//
        )));
        types.put(0, new ArrayList<>(Arrays.asList(
                FALSE, TRUE, TRUE, TRUE, FALSE,//
                FALSE, TRUE, FALSE, TRUE, FALSE,//
                FALSE, TRUE, FALSE, TRUE, FALSE,//
                FALSE, TRUE, TRUE, TRUE, FALSE//
        )));
        types.put(7, new ArrayList<>(Arrays.asList(
                FALSE, TRUE, TRUE, TRUE, TRUE,//
                FALSE, FALSE, FALSE, TRUE, FALSE,//
                FALSE, FALSE, TRUE, FALSE, FALSE,//
                FALSE, TRUE, FALSE, FALSE, FALSE//
        )));
        types.put(4, new ArrayList<>(Arrays.asList(
                FALSE, TRUE, FALSE, TRUE, FALSE,//
                FALSE, TRUE, TRUE, TRUE, FALSE,//
                FALSE, FALSE, FALSE, TRUE, FALSE,//
                FALSE, FALSE, FALSE, TRUE, FALSE//
        )));
    }

    @Override
    public List<List<Boolean>> getTrainingBatch(int batchSize) {
        List<List<Boolean>> batch = new ArrayList<>(batchSize);
        Collections.fill(batch,types.get(0));
        return batch;
    }


}
