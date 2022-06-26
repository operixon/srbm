package org.wit.snr.nn.srbm.visualization;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.stream.Collectors;

// auto update driven by matrix change
// time trigered update
// border for pixel, magnification of matrix visualization
public class ManyMatrixInFrame implements MatrixRendererIF {


    public static final int CELL_SIZE = 2;
    public static final int CELL_SPACE = 0;
    final JFrame frame; // TODO : refactor
    final List<Matrix> m;
    final private int x;
    final private int y;


    public ManyMatrixInFrame(List<Matrix> m) {
        frame = new JFrame("Many Matrix size:" + m.size());
        frame.setSize(700, 700);
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
        Container pane = frame.getContentPane();
        pane.setLayout(new GridLayout(20,20));
        List<MatrixPanel> collect = m.stream()
                .map(MatrixPanel::new)
               // .peek(p -> p.setAlignmentX(Component.CENTER_ALIGNMENT))
               // .peek(p->p.setSize(28,28))
                .peek(pane::add)
                .collect(Collectors.toList());
        frame.pack();
        frame.setVisible(true);
    }

    final private class MatrixPanel extends JPanel {

        Matrix matrix;

        public MatrixPanel(Matrix matrix) {
            this.matrix = matrix;
        }

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
            final List<List<Double>> data = matrix.normalize(0, 255).getMatrixAsCollection();
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
    }

    ;


}
