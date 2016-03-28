/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.minst;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/**
 * This class parse metadata and loads data from idx files. <br>
 * Detail format description : http://yann.lecun.com/exdb/mnist/<br>
 * This version loads all file data to memory.
 *
 * @author koperix
 */
public class IdxFileInMemory extends IdxFile {
   
    final String imagesPath;
    protected byte[] data;


    public IdxFileInMemory(String imagesPath) throws IOException {
        this.imagesPath = imagesPath;
    }

    @Override
    protected IdxFileMetadata readHeader() throws IOException {
        InputStream is = null;
        DataInputStream dis = null;
        try {
            // create input stream from file input stream
            is = new FileInputStream(imagesPath);
            // create data input stream
            dis = new DataInputStream(is);
            // Magic number
            // Two first bytes are 00
            dis.readByte();
            dis.readByte();
            // The third byte codes the type of the data:
            final int typeOfDataCode = dis.readUnsignedByte();
            // The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
            final int numberOfDimensions = dis.readByte();
            // The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
            final int[] sizeInDimension = new int[numberOfDimensions];
            for (int i = 0; i < sizeInDimension.length; i++) {
                sizeInDimension[i] = dis.readInt();
            }
            return new IdxFileMetadata(
                    MinstUtils.getTypeOfDataByCode(typeOfDataCode),
                    numberOfDimensions,
                    sizeInDimension
            );
        } finally {
            // releases any associated system files with this stream
            if (is != null) {
                is.close();
            }
            if (dis != null) {
                dis.close();
            }
        }

    }

    @Override
    protected void loadData() throws IOException {
        //TODO : is,dis one shot
        this.fileMetadata = readHeader();
        InputStream is = null;
        DataInputStream dis = null;
        try {
            // create input stream from file input stream
            is = new FileInputStream(imagesPath);
            // create data input stream
            dis = new DataInputStream(is);
            // Skip header
            dis.skipBytes(getHeaderLength());
            int dataLength = getDataLength();
            this.data = new byte[dataLength];
            dis.readFully(this.data);
            if (dis.available() > 0) {
                throw new IllegalStateException(
                        String.format(
                                "Ładowanie danych powinno wyczerpać wszystkie dane w pliku."
                                + "[ dataLength=%d; dis.available()=%d]",
                                dataLength,
                                dis.available())
                );
            }
        } catch (Exception e) {
            // if any I/O error occurs
            // TODO : logger
            e.printStackTrace();
        } finally {
            // releases any associated system files with this stream
            if (is != null) {
                is.close();
            }
            if (dis != null) {
                dis.close();
            }
        }

    }


    @Override
    public byte[] getData() throws IOException {
        loadData();
        if (fileMetadata == null) {
            throwMetadataFileNotLoadedException();
        }
        return data;
    }

    @Override
    public byte[] getData(int from, int to) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
