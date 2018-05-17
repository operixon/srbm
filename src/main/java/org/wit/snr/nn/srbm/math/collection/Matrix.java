package org.wit.snr.nn.srbm.math.collection;

import java.util.List;

public abstract class Matrix {


    public String toFullString() {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < getRowsNumber(); i++) {
            sb.append("| ");
            for (int j = 0; i < getColumnsNumber(); j++) {
                sb.append(String.format("%.2f", this.get(i, j)));
                sb.append("|");
            }
            sb.append(" |\n");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("{Matrix rows=%d,columns=%d}", getRowsNumber(), getColumnsNumber());
    }

    abstract public double get(int rowIndex, int columnIndex);

    abstract public void set(int rowIndex, int columnIndex, double value);

    public Matrix subtract(Matrix m) {
        assertEqualSize(m);
        Matrix result = instance(getRowsNumber(), getColumnsNumber());
        for (int i = 0; i < getRowsNumber(); i++) {
            for (int j = 0; j < getColumnsNumber(); j++) {
                result.set(i, j, this.get(i, j) - m.get(i, j));
            }
        }
        return result;
    }

    protected void assertEqualSize(Matrix m) {
        if (m.getRowsNumber() != getRowsNumber() && m.getColumnsNumber() != getColumnsNumber()) {
            throw new IllegalArgumentException("Substract is defined only for matrix's with the same size."
                    + "(" + getRowsNumber() + "x" + getColumnsNumber() + ") != (" + m.getRowsNumber() + "x" + m.getColumnsNumber() + ")");
        }
    }

    abstract public Matrix scalarDivide(double divVal);

    abstract public Matrix scalarMultiply(double mulVal);

    public Matrix matrixAdd(Matrix m) {
        assertEqualSize(m);
        Matrix result = instance(getRowsNumber(), getColumnsNumber());
        for (int i = 0; i < getRowsNumber(); i++) {
            for (int j = 0; j < getColumnsNumber(); j++) {
                result.set(i, j, this.get(i, j) + m.get(i, j));
            }
        }
        return result;
    }

    abstract public List<Double> getDataAsList();

    abstract public Matrix transpose();

    abstract protected Matrix instance(int rows, int columns);

    public Matrix multiplication(Matrix m) {
        assertSizeToMultiplication(this, m);
        Matrix result = instance(getRowsNumber(), m.getColumnsNumber());
        final int n = this.getColumnsNumber();
        for (int i = 0; i < result.getRowsNumber(); i++) {
            for (int j = 0; j < result.getColumnsNumber(); j++) {
                result.set(i, j, multiplyRowByColumn(n, i, j, this, m));
            }
        }
        return result;
    }

    double multiplyRowByColumn(int n, int i, int j, Matrix a, Matrix b) {
        double result = 0;
        for (int x = 0; x < n; x++) {
            result += a.get(i, n) * b.get(n, i);
        }
        return result;
    }

    void assertSizeToMultiplication(Matrix a, Matrix b) {
        if (a.getColumnsNumber() != b.getRowsNumber()) {
            throw new IllegalArgumentException(String.format("Matrix A columns not equals B rows. %s, %s", a, b));
        }
    }

    abstract public int getColumnsNumber();

    abstract public int getRowsNumber();

    abstract public Matrix gibsSampling();

    abstract public List<List<Double>> getMatrixAsCollection();

    /**
     * Suoming all cel values in each row.
     *
     * @return column vector expresed by Matrix object
     */
    public abstract Matrix rowsum();

    public abstract List<Double> getColumn(int columnIndex);
}
