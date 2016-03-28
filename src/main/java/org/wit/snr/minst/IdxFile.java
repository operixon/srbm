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
public abstract class IdxFile {
    
    protected IdxFileMetadata fileMetadata;
    
    protected abstract IdxFileMetadata readHeader() throws IOException;

    protected abstract void loadData() throws IOException;

    public abstract byte[] getData() throws IOException;
    
    public abstract byte[] getData(int from, int to);
    
    /**
     * Data length is product off all dimensions sizes and size of data unit
     *
     * @return
     */
    protected int getDataLength() {
        if (fileMetadata == null) {
            throwMetadataFileNotLoadedException();
        }
        final int[] sizeInDimension = fileMetadata.getSizeInDimension();
        final TypeOfData typeOfData = fileMetadata.getTypeOfData();
        int dataLength = sizeInDimension[0];
        for (int i = 1; i < sizeInDimension.length; i++) {
            dataLength *= sizeInDimension[i];
        }
        dataLength *= typeOfData.getNumOfBytes();
        return dataLength;
    }

    protected void throwMetadataFileNotLoadedException() throws IllegalStateException {
        throw new IllegalStateException("Nie pobrano jeszcze danych z nagłówka, albo coś poszło nie tak w trakcie ich pobierania.");
    }

    protected int getHeaderLength() {
        return // 2 <<Two zeros in magic number>> + 1 <<data code>> + 1 <<num of dims>>
        4 + // plus
        (fileMetadata.getNumberOfDimensions() * 4); // 4 byte int per dim :: info about size off a dim
    }

    public IdxFileMetadata getFileMetadata(){return fileMetadata;}

    
    
    
}
