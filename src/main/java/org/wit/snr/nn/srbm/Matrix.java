package org.wit.snr.nn.srbm;

import java.util.List;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class Matrix {

    final private List<List<Double>> data;
    final private int rows;
    final private int columns;

    public static Matrix createMatrixInitializedByRandomValues(final int w, final int h) {
        List<List<Double>> randomMatrix = SRBMUtils.getRandomMatrix(w, h);
        Matrix m = new Matrix(randomMatrix);
        return m;
    }

    public Matrix(List<List<Double>> data) {
        this.data = data;
        this.rows = data.size();
        this.columns = data.get(0).size(); // ? meaby make some validations / asserations
    }




    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        for (List<Double> row : data) {
            sb.append("| ").append(row.toString()).append(" |\n");
        }
        return sb.toString();
    }

    public Double get(int rowIndex, int columnIndex){
        return data.get(rowIndex).get(columnIndex);
    }

    public Matrix substract(Matrix m) {
        assertEqualSize(m);
        for(int rowIndex =0; rowIndex < rows; rowIndex++){
            for(int columnIndex =0; columnIndex < columns; columnIndex++){
                data.get(rowIndex).add(get(rowIndex,columnIndex) - m.get(rowIndex,columnIndex));
            }
        }
        return this;
    }

    private void assertEqualSize(Matrix m) {
        if (m.rows != rows && m.columns != columns) {
            throw new IllegalArgumentException("Substract is defined only for matrix's with the same size."
                    + "(" + rows + "x" + columns + ") != (" + m.rows + "x" + m.columns + ")");
        }
    }

    public Matrix scalarDivide(final double divVal) {
        data.stream().forEach(doubles -> doubles.forEach(aDouble -> aDouble = aDouble/divVal));
        return this;
    }

    public Matrix scalarMultiply(final double mulVal) {
        data.stream().forEach(doubles -> doubles.forEach(aDouble -> aDouble = aDouble*mulVal));
        return this;
    }

    public Matrix matrixAdd(Matrix m) {
        assertEqualSize(m);
        for(int rowIndex =0; rowIndex < rows; rowIndex++){
            for(int columnIndex =0; columnIndex < columns; columnIndex++){
                data.get(rowIndex).add(get(rowIndex,columnIndex) + m.get(rowIndex,columnIndex));
            }
        }
        return this;
    }

    public List<List<Double>> getData() {
        return data;
    }
}
