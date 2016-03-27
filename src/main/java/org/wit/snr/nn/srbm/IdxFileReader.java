/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/**
 *
 * @author koperix
 */
public class IdxFileReader {

    final String imagesPath;

    private int typeOfDataCode;
    private int numberOfDimensions;
    private int[] sizeInDimension;
    private int headerLength;
    private byte[] data;

    public IdxFileReader(String imagesPath) {
        this.imagesPath = imagesPath;
    }

    public void readHeader() throws IOException {
        InputStream is = null;
        DataInputStream dis = null;
        try {
            // create input stream from file input stream
            is = new FileInputStream(imagesPath);
            // create data input stream
            dis = new DataInputStream(is);

            // Magic number
            // Two first bytes are 00
            System.out.println("Magic number");
            dis.readByte();
            dis.readByte();
            // The third byte codes the type of the data:
            typeOfDataCode = dis.readUnsignedByte();
            // The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
            numberOfDimensions = dis.readByte();
            // The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
            sizeInDimension = new int[numberOfDimensions];
            for (int i = 0; i < sizeInDimension.length; i++) {
                sizeInDimension[i] = dis.readInt();
            }
            headerLength = computeHeaderLength();
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

    public void loadData() throws IOException {

        InputStream is = null;
        DataInputStream dis = null;
        try {
            // create input stream from file input stream
            is = new FileInputStream(imagesPath);
            // create data input stream
            dis = new DataInputStream(is);
            // Skip header
            dis.skipBytes(headerLength);
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

    /**
     * Data length is product off all dimensions sizes and size of data unit
     *
     * @return
     */
    private int getDataLength() {
        int dataLength = sizeInDimension[0];
        for (int i = 1; i < sizeInDimension.length; i++) {
            dataLength *= sizeInDimension[i];
        }
        dataLength *= MinstUtils.getTypeOfDataByCode(this.typeOfDataCode).getNumOfBytes();
        return dataLength;
    }

    private int computeHeaderLength() {
        return // 2 bytes :: Two zeros in magic number
                // 1 byte  :: data code
                // 1 byte  :: num of dims
                4
                + // 4 byte int per dim :: info about size off a dim
                (numberOfDimensions * 4);

    }
}
