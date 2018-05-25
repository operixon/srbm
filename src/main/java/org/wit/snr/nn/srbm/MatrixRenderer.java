package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.awt.*;
import java.util.List;

public class MatrixRenderer {

    final private int x;
    final private int y;
    final private Matrix m;
    final private Graphics g;

    public MatrixRenderer(int x, int y, Matrix m, Graphics g) {
        this.x = x;
        this.y = y;
        this.m = m;
        this.g = g;
    }

    public void render() {
        int colidx = 0;
        for (List<Double> column : m.getMatrixAsCollection()) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    double color = column.get(i * 28 + j);
                    int ic = (int) Math.round(color);
                    g.setColor(new Color(ic));
                    int offset_i = x + 28 * (colidx % 20) + 5 * (colidx % 20);
                    int offset_j = y + 28 * (Math.round(colidx / 20)) + 5 * (Math.round(colidx / 20));
                    g.drawLine(i + offset_i, j + offset_j, i + offset_i, j + offset_j);
                }
            }
            colidx++;
        }
    }

}
