package org.wit.snr.nn.srbm.math.collection;

import java.text.MessageFormat;
import java.util.HashMap;
import java.util.Map;

/**
 * @param <X>
 */
public class SparseMatrix<X> {

    private Map<String , X> values = new HashMap<>();

    private final int rows;

    private final int cols;

    public SparseMatrix(int cols, int rows) {
        this.rows = rows;
        this.cols = cols;
    }

    // in future meaby use cantor function to maping x y pair to single number
    public X get(int x, int y) {
        validateMatrixIndex(x,y);
        return values.get(x+":"+y);
    }

    public void set(int x, int y, X value) {
        validateMatrixIndex(x,y);
        values.put(x+":"+y, value);
    }

    private void validateMatrixIndex(int x, int y){
        if(x+1 > cols || y+1 > rows){
            throw new IndexOutOfBoundsException (MessageFormat.format("Matrix dimension {0}x{1}. Get index {2}x{3}", this.cols, this.rows, x, y));
        }
    }

    /**
     *
     * @return matrix fill
     */
    public double getSparsnesFactor(){
        return (double) values.size() / (((double)rows * (double)cols) / 100.0);
    }

}