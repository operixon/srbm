package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;

/**
 * Created by akoperkiewicz on 14.01.2017.
 */
public interface TrainingSet {

    Matrix getTrainingBatch(int batchSize);

}
