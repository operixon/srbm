/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

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
public class MinstLabelLoader {

    String imagesPath = "/home/koperix/Pobrane/minst/t10k-images-idx3-ubyte";

    public void load() throws IOException {

        InputStream is = null;
        DataInputStream dis = null;
        try {
            // create input stream from file input stream
            is = new FileInputStream(imagesPath);
            // create data input stream
            dis = new DataInputStream(is);
            // count the available bytes form the input stream

            // Magic number
            // Two first bytes are 00
            System.out.println("Magic number");
            dis.readByte();
            dis.readByte();
            System.out.println(String.format("Type off data: %02x", dis.readByte()));
            int numOffDims = dis.readByte();
            System.out.println(String.format("Number off dimensions: %02d ", numOffDims));

            int[] dimensions = new int[numOffDims];
            int pixelDataToRead = 1;
            for (int i = 0; i < dimensions.length; i++) {
                dimensions[i] = dis.readInt();
                pixelDataToRead *= dimensions[i];
            }
            System.out.println("Dimensions: " + Arrays.toString(dimensions));
            System.out.println("Pixel data to read: " + pixelDataToRead);
            // Reading pixel data
            byte[][][] pixelData = new byte[10000][28][28];
            for(int i=0;i<10000;i++){
                for(int j=0;j<28;j++){
                 dis.read(pixelData[i][j]);
                }
            }
            System.out.println("EOF ? :" + dis.available());

            for(int i =0;i<10;i++){
                for(int r=0;r <28;r++){
                    for(int c=0;c<28;c++){
                        System.out.print((pixelData[i][r][c]& 0xff)>150?"O":" ");
                    }
                            System.out.println("");
                }
                System.out.println("\n============================\n");
            }
            
            
        } catch (Exception e) {
            // if any I/O error occurs
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

}
