package org.wit.snr.nn.srbm.math.collection;

import java.util.List;

public abstract class Matrix {


    public String toFullString() {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < getRows(); i++) {
            sb.append("| ");
            for (int j = 0; i < getColumns(); j++) {
                sb.append(String.format("%.2f", this.get(i, j)));
                sb.append("|");
            }
            sb.append(" |\n");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("{Matrix rows=%d,columns=%d}", getRows(), getColumns());
    }

    abstract public double get(int rowIndex, int columnIndex);

    abstract public void set(int rowIndex, int columnIndex, double value);

    public Matrix subtract(Matrix m) {
        assertEqualSize(m);
        Matrix result = instance(getRows(), getColumns());
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getColumns(); j++) {
                result.set(i, j, this.get(i, j) - m.get(i, j));
            }
        }
        return result;
    }

    protected void assertEqualSize(Matrix m) {
        if (m.getRows() != getRows() && m.getColumns() != getColumns()) {
            throw new IllegalArgumentException("Substract is defined only for matrix's with the same size."
                    + "(" + getRows() + "x" + getColumns() + ") != (" + m.getRows() + "x" + m.getColumns() + ")");
        }
    }

    abstract public Matrix scalarDivide(double divVal);

    abstract public Matrix scalarMultiply(double mulVal);

    public Matrix matrixAdd(Matrix m) {
        assertEqualSize(m);
        Matrix result = instance(getRows(), getColumns());
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getColumns(); j++) {
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
        Matrix result = instance(getRows(), m.getColumns());
        final int n = this.getColumns();
        for (int i = 0; i < result.getRows(); i++) {
            for (int j = 0; j < result.getColumns(); j++) {
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
        if (a.getColumns() != b.getRows()) {
            throw new IllegalArgumentException(String.format("Matrix A columns not equals B rows. %s, %s", a, b));
        }
    }

    abstract public int getColumns();

    abstract public int getRows();

    abstract public Matrix gibsSampling();

    abstract public List<List<Double>> getMatrixAsCollection();

    public abstract Matrix rowsum();
}
