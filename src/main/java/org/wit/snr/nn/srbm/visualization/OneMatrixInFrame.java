package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.util.List;

// auto update driven by matrix change
// time trigered update
// border for pixel, magnification of matrix visualization
public class OneMatrixInFrame implements MatrixRendererIF {


    public static final int CELL_SIZE = 2;
    public static final int CELL_SPACE = 1;
    final JFrame frame; // TODO : refactor
    final Matrix m;
    final private int x;
    final private int y;


    public OneMatrixInFrame(String title, Matrix m) {
        frame = new JFrame("Matrix rows:" + m.getRowsNumber() + ", cols: " + m.getColumnsNumber());
        //int width = m.getColumnsNumber() * CELL_SIZE + m.getColumnsNumber() * CELL_SPACE;
       // int height = m.getRowsNumber() * CELL_SIZE + m.getRowsNumber() * CELL_SPACE;
        frame.setSize(500, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setResizable(true);
        frame.setTitle(title);
     frame.setBackground(Color.BLACK);
        x = 0;
        y = 0;
        this.m = m;
    }


    @Override
    public void render() {
        paintPanel.setBackground(Color.BLACK);
        frame.add(paintPanel);
        frame.setVisible(true);
    }

    final JPanel paintPanel = new JPanel() {
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.setColor(Color.BLACK);
            g.fillRoundRect(
                    0,
                    0,
                    2000,
                    2000,
                    0,
                    0);
            final List<List<Double>> data = m.normalize(0, 255).getMatrixAsCollection();
            final int cols = data.size();
            for (int col = 0; col < cols; col++) {
                for (int row = 0; row < data.get(col).size(); row++) {
                    double cell = data.get(col).get(row);
                    int c = (int) Math.round(cell);
                    g.setColor(new Color(c, c, c));
                    int pixelX = row * CELL_SIZE + row * CELL_SPACE + y;
                    int pixelY = col * CELL_SIZE + col * CELL_SPACE + x;
                    g.fillRoundRect(
                            pixelX,
                            pixelY,
                            CELL_SIZE,
                            CELL_SIZE,
                            0,
                            0);
                }
            }
        }
    };


}
