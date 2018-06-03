package org.wit.snr.nn.srbm.math.collection;

import org.ojalgo.matrix.BasicMatrix;
import org.ojalgo.matrix.PrimitiveMatrix;

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


    private PrimitiveMatrix exportToAjo() {
        BasicMatrix.Factory<PrimitiveMatrix> matrixFactory = PrimitiveMatrix.FACTORY;
        BasicMatrix.Builder<PrimitiveMatrix> builder
                = matrixFactory.getBuilder(this.getRowsNumber(), this.getColumnsNumber());

        for (int r = 0; r < getRowsNumber(); r++) {
            for (int c = 0; c < getColumnsNumber(); c++) {
                builder.set(r, c, get(r, c));
            }
        }
        return builder.build();
    }

    public Matrix multiplication(Matrix m) {
        PrimitiveMatrix A = exportToAjo();
        PrimitiveMatrix B = m.exportToAjo();
        PrimitiveMatrix multiply = A.multiply(B);
        Matrix result = instance((int) multiply.countRows(), (int) multiply.countColumns());
        for (int r = 0; r < result.getRowsNumber(); r++) {
            for (int c = 0; c < result.getColumnsNumber(); c++) {
                result.set(r, c, multiply.get(r, c));
            }
        }
        return result;

    }

    double multiplyRowByColumn(int multiplVectorSize, int i, int j, Matrix a, Matrix b) {
        double result = 0;
        for (int x = 0; x < multiplVectorSize; x++) {
            result += a.get(i, x) * b.get(x, j);
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

    public abstract Matrix normalize(int from, int to);

    public abstract Matrix reshape(int columnLength);


}
