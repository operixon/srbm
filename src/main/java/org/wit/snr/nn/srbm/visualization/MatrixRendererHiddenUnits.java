package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.awt.*;
import java.util.List;

public class MatrixRendererHiddenUnits {

    final private int x;
    final private int y;
    final private Matrix m;
    final private Graphics g;

    public MatrixRendererHiddenUnits(int x, int y, Matrix m, Graphics g) {
        this.x = x;
        this.y = y;
        this.m = m;
        this.g = g;
    }

    public void render() {
        final int cellSize = 8;
        final List<List<Double>> data = m.normalize(0, 255).getMatrixAsCollection();
        final int cols = data.size();
        final int rows = data.get(0).size();
        final int cellSpace = 1;
        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < data.get(col).size(); row++) {
                double cell = data.get(col).get(row);
                int c = (int) Math.round(cell);
                g.setColor(new Color(c, c, c));
                g.fillRoundRect(
                        row * cellSize + row * cellSpace + y,
                        col * cellSize + col * cellSpace + x,
                        cellSize,
                        cellSize,
                        0,
                        0);
            }
        }
    }

}
