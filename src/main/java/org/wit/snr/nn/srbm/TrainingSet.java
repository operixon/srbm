package org.wit.snr.nn.srbm;

import java.util.BitSet;
import java.util.List;

/**
 * Created by akoperkiewicz on 14.01.2017.
 */
public interface TrainingSet<V> {

    List<V> getTrainingBatch(int batchSize);

}
