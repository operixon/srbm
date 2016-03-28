/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.minst;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/**
 *
 * @author koperix
 */
public class MinstImageLoader extends IdxFileInMemory {

    private int[][][] images;

    public MinstImageLoader(String imagesPath) throws IOException {
        super(imagesPath);
        loadImages();
    }

    private void loadImages() throws IOException {
        final byte[] d = super.getData();
        final int numImg = getNumberOfImages();
        final int numCol = getNumberOfColumns();
        final int numRow = getNumberOfRows();
        final int bytesPerImg = getBytesPerImage();
        final int bytesPerRow = getBytesPerRow();
        images = new int[numImg][numRow][numCol];
        for (int i = 0; i < numImg; i++) {
            for (int r = 0; r < numRow; r++) {
                int offset = i * bytesPerImg + r * bytesPerRow;
                System.arraycopy(d, offset, images[i][r], 0, bytesPerRow);
            }
        }

    }

    public int getNumberOfImages() {
        return fileMetadata.getSizeInDimension()[0];
    }

    public int getNumberOfRows() {
        return fileMetadata.getSizeInDimension()[1];
    }

    public int getNumberOfColumns() {
        return fileMetadata.getSizeInDimension()[2];
    }

    private int getBytesPerColumn() {
        return fileMetadata.getTypeOfData().getNumOfBytes() * getNumberOfRows();
    }

    private int getBytesPerRow() {
        return fileMetadata.getTypeOfData().getNumOfBytes() * getNumberOfColumns();
    }

    private int getBytesPerImage() {
        return fileMetadata.getTypeOfData().getNumOfBytes() * getNumberOfColumns() * getNumberOfRows();
    }

    public int[][] getImage(int imageIdx) {
        return images[imageIdx];
    }

    public int[][][] getImages() {
        return images;
    }

}
