package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.awt.*;
import java.util.List;

public class MatrixRendererSample implements MatrixRendererIF {

    final private int x;
    final private int y;
    final private Matrix m;
    final private Graphics g;
    final Color color;

    public MatrixRendererSample(int x, int y, Matrix m, Graphics g, Color color) {
        this.x = x;
        this.y = y;
        this.m = m;
        this.g = g;
        this.color = color;
    }

    @Override
    public void render() {

        int colidx = 0;
        for (List<Double> column : m.getMatrixAsCollection()) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    double matrixVals = column.get(i * 28 + j);
                    g.setColor(color);
                    int offset_i = x + 28 * (colidx % 20) + 2;
                    int offset_j = y + 28 * (Math.round(colidx / 20)) + 2;
                    if (matrixVals > 0) g.drawLine(i + offset_i, j + offset_j, i + offset_i, j + offset_j);
                }
            }
            colidx++;
        }
    }

}
