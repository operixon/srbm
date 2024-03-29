/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.minst;

import java.io.IOException;

/**
 *
 * @author koperix
 */
public class MinstImageLoader extends IdxFileInMemory {

    private byte[][][] images;

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
        images = new byte[numImg][numRow][numCol];
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

    public byte[][] getImage(int imageIdx) {
        return images[imageIdx];
    }

    public byte[][][] getImages() {
        return images;
    }

}
