package org.wit.snr.nn.srbm.math.collection;

import org.wit.snr.nn.srbm.math.MathUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class Matrix2D extends Matrix {

    public static Matrix createMatrixWithRandomValues(int rows, int columns) {
        List<List<Double>> matrixData = MathUtils.getRandomMatrix(rows, columns);
        return new Matrix2D(matrixData);
    }


    public static Matrix createFilledMatrix(int rows, int columns, double fillValue) {
        List<List<Double>> columnsList = new ArrayList<>(columns);
        for (int i = 0; i < columns; i++) {
            List<Double> column = new ArrayList<>(rows);
            for (int j = 0; j < rows; j++) {
                column.add(fillValue);
            }
            columnsList.add(column);
        }
        return new Matrix2D(columnsList);
    }

    public static Matrix createColumnVector(List<Double> vectorData) {
        List<List<Double>> reply = new ArrayList<>(1);
        reply.add(vectorData);
        return new Matrix2D(reply);
    }

    final private List<List<Double>> columnsList;
    final int columns;
    final int rows;
    final private static Random random = new Random();


    public Matrix2D(List<List<Double>> columnsList) {
        if (columnsList == null) {
            throw new NullPointerException();
        }
        this.columnsList = columnsList;
        this.columns = columnsList.size();
        this.rows = columnsList.get(0).size();
    }

    @Override
    public double get(int rowIndex, int columnIndex) {
        return columnsList.get(columnIndex).get(rowIndex);
    }


    @Override
    public void set(int rowIndex, int columnIndex, double value) {

        columnsList.get(columnIndex).set(rowIndex, value);
    }


    @Override
    public Matrix scalarDivide(final double divVal) {
        List<List<Double>> collect = columnsList.stream().map(column -> column.stream().map(cel -> cel = cel / divVal).collect(toList())).collect(toList());

        return new Matrix2D(collect);
    }

    @Override
    public Matrix scalarMultiply(final double mulVal) {

        List<List<Double>> collect = columnsList.stream().map(column -> column.stream().map(cel -> cel = cel * mulVal).collect(toList())).collect(toList());

        return new Matrix2D(collect);
    }


    @Override
    public List<Double> getDataAsList() {

        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix transpose() {
        Matrix result = instance(getColumnsNumber(), getRowsNumber());
        for (int r = 0; r < result.getRowsNumber(); r++) {
            for (int c = 0; c < result.getColumnsNumber(); c++) {
                result.set(r, c, get(c, r));
            }
        }
        return result;
    }

    @Override
    protected Matrix instance(int rows, int columns) {
        return createFilledMatrix(rows, columns, 0.0);
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
        List<List<Double>> sampledMatrixData = columnsList
                .stream()
                .map(
                row -> row
                        .stream()
                        .map(cel -> cel > random.nextDouble() ? 1.0 : 0.0)
                        .collect(toList())
        ).collect(toList());
        return new Matrix2D(sampledMatrixData);
    }

    @Override
    public List<List<Double>> getMatrixAsCollection() {
        return columnsList;
    }

    @Override
    public Matrix rowsum() {
        List<List<Double>> rowsList = getRowsList();
        List<Double> rowsum = rowsList.stream()
                .map(row -> row.stream().collect(Collectors.summingDouble(Double::doubleValue)))
                .collect(Collectors.toList());
        return createColumnVector(rowsum);
    }

    @Override
    public List<Double> getColumn(int columnIndex) {
        return columnsList.get(columnIndex);
    }

    @Override
    public Matrix normalize(int to) {
        final double max = getMaxValue();
        final double min = getMinValue();
        final double range = max - min;
        List<List<Double>> collect = this.columnsList.stream()
                .map(
                        column -> column.stream()
                                .map(cel -> to * ((cel - min) / range))
                                .collect(toList()))
                .collect(toList());
        return new Matrix2D(collect);
    }

    private double getMinValue() {
        return 0;
    }

    private double getMaxValue() {
        return 0;
    }

    private List<List<Double>> getRowsList() {
        return this.transpose().getMatrixAsCollection();
    }

}
