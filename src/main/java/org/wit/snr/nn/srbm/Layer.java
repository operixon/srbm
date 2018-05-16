package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.util.Collections;
import java.util.List;

public class Layer {

    public final Matrix W;
    public final List<Double> vbias; // ci
    public final List<Double> hbias; // bj

    public final int inputSize;
    public final int outputSize;


    public Layer(int numdims, int numhid) {
        W = Matrix2D.createMatrixWithRandomValues(numdims, numhid);
        vbias = Collections.nCopies(numdims,1.0);
        hbias = Collections.nCopies(numhid,1.0);
        inputSize = numdims;
        outputSize = numhid;
    }

    /**
     * This method compute activation signal for one neuron
     * from hidden layer. This value is defined as a
     * SUM_i( W_ij * V_i ).
     * This computation is related with positive phase.
     *
     * @param j index of unit from hidden layer
     * @return activation signal value for #hiddenUnitIndex
     */
    public Double getWeightsSumForHiddenUnit(List<Double> visibleUnits, final int j) {
        double sum = 0;
        for(int i = 0; i < inputSize; i++ ){
            final double unit = visibleUnits.get(i);
            if (unit != 1.0 && unit != 0.0) {
                throw new IllegalStateException(String.format("Unit value schould be 1 or 0. But found %s", unit));
            }
            sum += W.get(j, i) * unit;
        }
        return sum;
    }

    /**
     * This method compute activation signal for one neuron
     * from visible layer. This value is defined as a
     * SUM_j( W_ij * H_j )
     * This computation is related with negative phase.
     *
     * @param i index of unit from visible layer
     * @return activation signal value for i neuron from visible layer
     */
    public Double getWeightsSumForVisibleUnit(List<Double> hiddenUnits, final int i) {
        double sum = 0;
        for(int j = 0; j < outputSize; j++ ){
            final double unit = hiddenUnits.get(i);
            if (unit != 1.0 && unit != 0.0) {
                throw new IllegalStateException(String.format("Unit value schould be 1 or 0. But found %s", unit));
            }
            sum += W.get(j, i) * unit;
        }
        return sum;
    }

    /**
     * Computes weights summary plus bias.
     * Z = B_j + SUM_i (W_ij * V_i)
     * Positive phase.
     *
     * @param visibleUnits
     * @param j
     * @return
     */
    public Double getActivationSignalForHiddenUnit(List<Double> visibleUnits, final int j) {
        return hbias.get(j)  + getWeightsSumForHiddenUnit(visibleUnits, j);

    }

    /**
     * Compute summary plus bias.
     * Z = C_i + SUM_j (W_ij * H_j)
     * Negative phase.
     *
     * @param visibleUnits
     * @return
     */
    public Double getActivationSignalForVisibleUnit(List<Double> visibleUnits, final int i) {
        return vbias.get(i) + getWeightsSumForVisibleUnit(visibleUnits,i);
    }


}
