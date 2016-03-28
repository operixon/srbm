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
        byte[] d = super.getData();
        images = new int[getNumberOfImages()][getNumberOfRows()][getNumberOfColumns()];
        for (int i = 0; i < getNumberOfImages(); i++) {
            for (int r = 0; r < getNumberOfRows(); r++) {
                for (int c = 0; c < getNumberOfColumns(); c++) {
                    int offset = i * getBytesPerImage() + r * getBytesPerRow();
                    System.arraycopy(d, offset, images[i][r],0,images[i][r].length);
                }
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

    private  int getBytesPerRow() {
        return fileMetadata.getTypeOfData().getNumOfBytes() * getNumberOfRows();
    }

    private  int getBytesPerColumn() {
        return fileMetadata.getTypeOfData().getNumOfBytes() * getNumberOfColumns();
    }

    private  int getBytesPerImage() {
        return fileMetadata.getTypeOfData().getNumOfBytes() * getNumberOfColumns() * getNumberOfRows();
    }

    public int[][] getImage(int imageIdx) {
        return images[imageIdx];
    }

    public int[][][] getImages() {
        return images;
    }

}