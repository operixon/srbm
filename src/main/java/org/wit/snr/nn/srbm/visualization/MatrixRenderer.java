package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.awt.*;
import java.util.List;

public class MatrixRenderer implements MatrixRendererIF{

    final private int x;
    final private int y;
    final private Matrix m;
    final private int w;
    final private Graphics g;

    public MatrixRenderer(int x, int y, int w, Matrix m, Graphics g) {
        this.x = x;
        this.y = y;
        this.m = m;
        this.g = g;
        this.w = w;
    }

    public void render() {
        int colidx = 0;
        for (List<Double> column : m.normalize(0, 255).getMatrixAsCollection()) {
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < w; j++) {
                    int c = (int) Math.round(column.get(i * w + j));
                    g.setColor(new Color(c, c, c));
                    int offset_i = x + w * (colidx % 20) + 2 * (colidx % 20);
                    int offset_j = y + w * (Math.round(colidx / 20)) + 2 * (Math.round(colidx / 20));
                    g.drawLine(i + offset_i, j + offset_j, i + offset_i, j + offset_j);
                }
            }
            colidx++;
        }
    }

}
