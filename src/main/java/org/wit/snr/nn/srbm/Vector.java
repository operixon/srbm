package org.wit.snr.nn.srbm;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class Vector<T> {

    final private List<T> data;

    public Vector(List<T> data) {
        this.data = data;
    }

    public Matrix multiplyByTransposedVector(Vector transposedVector) {
        int columns = data.size();
        int rows = transposedVector.size();
        List<List<Double>> result = new ArrayList<>(rows);
        for(int rowIndex =0 ; rowIndex < rows; rowIndex ++){
            List<Double> row = new ArrayList<>(columns);
            for(int columnIndex =0; columnIndex < columns;columnIndex++){
                double cellValue = multiply(data.get(columnIndex), transposedVector.data.get(rowIndex));
                row.add(columnIndex,cellValue);
            }
            result.add(rowIndex,row);
        }
        return new Matrix(result);
    }

    private double multiply(Object o1, Object o2) {
        if(o1 instanceof Double && o2 instanceof Boolean){
            return ((Double)o1) * (o2 == Boolean.TRUE ? 1.0 : 0.0);
        } else if (o1 instanceof Double && o2 instanceof Double) {
            return (Double)o1*(Double)o2;
        } else {
            throw new IllegalArgumentException("Undefined operation for types "+o1.getClass().getName()+", "+o2.getClass().getName());
        }
    }

    private int size() {
        return data.size();
    }
}
