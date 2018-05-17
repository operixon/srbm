package org.wit.snr.nn.srbm.math.collection;

import org.wit.snr.nn.srbm.math.MathUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class Matrix1D extends Matrix {

    static Matrix createMatrixWithRandomValues(int rows, int columns) {
        List<Double> matrixData = MathUtils.getRandomList(rows * columns);
        return new Matrix1D(matrixData, rows, columns);
    }

    static Matrix createMatrixFromArray(double[][] m) {
        final List<Double> d = new ArrayList<>();
        for (double[] mRow : m) {
            for (double cel : mRow) {
                d.add(cel);
            }
        }
        final int rows = m.length;
        final int columns = m[0].length;
        return new Matrix1D(d, rows, columns);
    }

    final private List<Double> data;
    private int columns;
    private int rows;


    public Matrix1D(List<Double> data, int rows, int columns) {
        if (data == null) {
            throw new NullPointerException();
        }
        if (columns * rows != data.size()) {
            throw new IllegalArgumentException(String.format("Matrix init size error. Width * Height != data.size(). rows=%d, columns=%d, data.size=%d.", rows, columns, data.size()));
        }
        this.data = data;
        this.rows = rows;
        this.columns = columns;
    }

    public Matrix1D(int rows, int columns) {
        this.data = new ArrayList<Double>(rows * columns);
        this.rows = columns;
        this.columns = rows;
    }


    @Override
    public double get(int rowIndex, int columnIndex) {
        return data.get(getFlatIndex(rowIndex, columnIndex));
    }

    private int getFlatIndex(int rowIndex, int columnIndex) {
        return rowIndex * columns + columnIndex;
    }

    @Override
    public void set(int rowIndex, int columnIndex, double value) {
        data.set(getFlatIndex(rowIndex, columnIndex), value);
    }


    @Override
    public Matrix scalarDivide(final double divVal) {
        data.stream().forEach(cel -> cel = cel / divVal);
        return this;
    }

    @Override
    public Matrix scalarMultiply(final double mulVal) {
        data.stream().forEach(aDouble -> aDouble = aDouble * mulVal);
        return this;
    }

    @Override
    public List<Double> getDataAsList() {
        return data;
    }

    @Override
    public Matrix transpose() {
        int oldRows = rows;
        int oldColumns = columns;
        rows = oldColumns;
        columns = oldRows;
        return this;
    }

    @Override
    protected Matrix instance(int rows, int columns) {
        return createMatrixWithRandomValues(rows, columns);
    }

    @Override
    public int getColumnsNumber() {
        return columns;
    }

    @Override
    public int getRowsNumber() {
        return rows;
    }

    @Override
    public Matrix gibsSampling() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<List<Double>> getMatrixAsCollection() {
        return null;
    }

    @Override
    public Matrix rowsum() {
        return null;
    }
}
