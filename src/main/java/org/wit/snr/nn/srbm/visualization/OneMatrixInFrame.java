package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.util.List;

// auto update driven by matrix change
// time trigered update
// border for pixel, magnification of matrix visualization
public class OneMatrixInFrame implements MatrixRendererIF{


    public static final int CELL_SIZE = 2;
    public static final int CELL_SPACE = 1;
    final JFrame frame; // TODO : refactor
    final Canvas canvas; // TODO : refacotr
    final Matrix m;
    final private int x;
    final private int y;

    public OneMatrixInFrame(Matrix m) {
        frame = new JFrame("Matrix rows:" + m.getRowsNumber() + ", cols: " + m.getColumnsNumber());
        frame.setSize(m.getColumnsNumber(), m.getRowsNumber());
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setResizable(true);
        frame.setVisible(true);
        //Creating the canvas.
        canvas = new Canvas();
        canvas.setSize(700, 500);
        canvas.setBackground(Color.BLACK);
        canvas.setVisible(true);
        canvas.setFocusable(false);
        //Putting it all together.
        frame.add(canvas);
        canvas.createBufferStrategy(3);
        x = 0;
        y = 0;
        this.m = m;
    }


    @Override
    public void render() {
        synchronized (canvas) {
            BufferStrategy bufferStrategy = canvas.getBufferStrategy();
            Graphics graphics = bufferStrategy.getDrawGraphics();
            renderMatrix(graphics, x, y);
            bufferStrategy.show();
            graphics.dispose();
        }
    }

    private void renderMatrix(Graphics g, int x, int y) {
        final List<List<Double>> data = m.normalize(0, 255).getMatrixAsCollection();
        final int cols = data.size();
        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < data.get(col).size(); row++) {
                double cell = data.get(col).get(row);
                int c = (int) Math.round(cell);
                g.setColor(new Color(c, 0, c));
                g.fillRoundRect(
                        row * CELL_SIZE + row * CELL_SPACE + y,
                        col * CELL_SIZE + col * CELL_SPACE + x,
                        CELL_SIZE,
                        CELL_SIZE,
                        0,
                        0);
            }
        }
    }

}
