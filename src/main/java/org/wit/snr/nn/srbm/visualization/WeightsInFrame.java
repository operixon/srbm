package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import javax.swing.*;
import java.awt.*;
import java.util.List;

// auto update driven by matrix change
// time trigered update
// border for pixel, magnification of matrix visualization
public class WeightsInFrame implements MatrixRendererIF {


    public static final int CELL_SIZE = 1;
    public static final int CELL_SPACE = 0;
    final JFrame frame; // TODO : refactor
    Matrix m;
    final private int x;
    final private int y;


    public WeightsInFrame(String title, Matrix m) {
        frame = new JFrame("Matrix rows:" + m.getRowsNumber() + ", cols: " + m.getColumnsNumber());
        //int width = m.getColumnsNumber() * CELL_SIZE + m.getColumnsNumber() * CELL_SPACE;
       // int height = m.getRowsNumber() * CELL_SIZE + m.getRowsNumber() * CELL_SPACE;
        frame.setSize(500, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
   frame.setLocationRelativeTo(null);
        frame.setResizable(true);
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

    public void render(Matrix w){
        this.m = w;
        render();
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
            int colidx = 0;
            for (List<Double> column : m.normalize(0, 255).getMatrixAsCollection()) {
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        int c = (int) Math.round(column.get(i * 28 + j));
                        g.setColor(new Color(c, c, c));
                        int offset_i = x + 28 * (colidx % 20) + 2 * (colidx % 20);
                        int offset_j = y + 28 * (Math.round(colidx / 20)) + 2 * (Math.round(colidx / 20));
                        g.drawLine(i + offset_i, j + offset_j, i + offset_i, j + offset_j);
                    }
                }
                colidx++;
            }
        }
    };


}
