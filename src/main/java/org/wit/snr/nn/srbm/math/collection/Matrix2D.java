package org.wit.snr.nn.srbm.math.collection;

import org.wit.snr.nn.srbm.math.MathUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.util.stream.Collectors.toList;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class Matrix2D extends Matrix {

    public static Matrix createMatrixWithRandomValues(int rows, int columns) {
        List<List<Double>> matrixData = MathUtils.getRandomMatrix(rows, columns);
        return new Matrix2D(matrixData);
    }

    public static Matrix createMatrixFromArray(double[][] m) {
        final List<List<Double>> d = new ArrayList<>();
        for (double[] mRow : m) {
            List<Double> row = new ArrayList<>();
            for (double cel : mRow) {
                row.add(cel);
            }
            d.add(row);
        }
        return new Matrix2D(d);
    }

    public static Matrix createFilledMatrix(int rows, int columns, double fillValue) {
        List<List<Double>> table = new ArrayList<>(columns);
        for (int i = 0; i < rows; i++) {
            List<Double> row = new ArrayList<>(rows);
            for (int j = 0; i < columns; j++) {
                row.add(fillValue);
            }
            table.add(row);
        }
        return new Matrix2D(table);
    }

    final private List<List<Double>> data;
    final int columns;
    final int rows;
    final private static Random random = new Random();


    public Matrix2D(List<List<Double>> data) {
        if (data == null) {
            throw new NullPointerException();
        }
        this.data = data;
        this.rows = data.size();
        this.columns = data.get(0).size();
    }

    @Override
    public double get(int rowIndex, int columnIndex) {
        return data.get(rowIndex).get(columnIndex);
    }


    @Override
    public void set(int rowIndex, int columnIndex, double value) {
        data.get(rowIndex).set(columnIndex, value);
    }


    @Override
    public Matrix scalarDivide(final double divVal) {
        data.stream().forEach(row -> row.stream().forEach(cel -> cel = cel / divVal));
        return this;
    }

    @Override
    public Matrix scalarMultiply(final double mulVal) {
        data.stream().forEach(row -> row.stream().forEach(cel -> cel = cel * mulVal));
        return this;
    }

    @Override
    public Matrix matrixAdd(Matrix m) {
        assertEqualSize(m);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.set(i, j, this.get(i, j) + m.get(i, j));
            }
        }
        return this;
    }

    @Override
    public List<Double> getDataAsList() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected Matrix instance(int rows, int columns) {
        return createFilledMatrix(rows, columns, 0.0);
    }

    @Override
    public int getColumns() {
        return columns;
    }

    @Override
    public int getRows() {
        return rows;
    }

    @Override
    public Matrix gibsSampling() {
        List<List<Double>> sampledMatrixData = data.stream().map(
                row -> row
                        .stream()
                        .map(cel -> cel > random.nextDouble() ? 1.0 : 0.0)
                        .collect(toList())
        ).collect(toList());
        return new Matrix2D(sampledMatrixData);
    }

    @Override
    public List<List<Double>> getMatrixAsCollection() {
        return data;
    }

}
