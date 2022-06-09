package org.wit.snr.nn.srbm.layer;

import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;

import java.util.List;

public class Layer {
    // W := randn(numdims, numhid)
    public Matrix W;
    public Matrix vbias; // ci
    public Matrix hbias; // bj

    public final int inputSize;
    public final int outputSize;
    public double error= 1;


    public Layer(int numdims, int numhid) {
        W = Matrix2D.createMatrixWithRandomValues(numdims, numhid);
        vbias = Matrix2D.createMatrixWithRandomValues(numdims, 1);
        hbias = Matrix2D.createMatrixWithRandomValues(numhid, 1);
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
            //   if (unit != 1.0 && unit != 0.0) {
            //       throw new IllegalStateException(String.format("Unit value schould be 1 or 0. But found %s", unit));
            //  }
            sum += W.get(i, j) * unit;
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
            final double hj = hiddenUnits.get(j);
          //  if (hj != 1.0 && hj != 0.0) {
          //      throw new IllegalStateException(String.format("Unit value schould be 1 or 0. But found %s", hj));
          //  }
            sum += W.get(i, j) * hj;
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
        return hbias.get(j, 0) + getWeightsSumForHiddenUnit(visibleUnits, j);

    }

    /**
     * Compute summary plus bias.
     * Zi = C_i + SUM_j (W_ij * H_j)
     * Negative phase.
     *
     * @param hiddenUnits
     * @return
     */
    public Double getActivationSignalForVisibleUnit(List<Double> hiddenUnits, final int i) {
        Double weightsSumForVisibleUnit = getWeightsSumForVisibleUnit(hiddenUnits, i);
        double v = vbias.get(i, 0);
        return v + weightsSumForVisibleUnit;
    }


}
