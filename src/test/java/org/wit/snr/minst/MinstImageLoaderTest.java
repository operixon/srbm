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
import static org.testng.Assert.*;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 *
 * @author koperix
 */
public class MinstImageLoaderTest {


    final static int NUMBER_OF_IMAGES = 10000;
    final static int IMAGE_ROWS = 28;
    final static int IMAGE_COLS = 28;

    MinstImageLoader mil ;

    @BeforeClass
    public void setUpClass() throws Exception {
        String f = getClass().getClassLoader().getResource("t10k-images-idx3-ubyte").getPath();
        mil = new MinstImageLoader(f);
        mil.loadData();
    }


    @Test
    public void testGetNumberOfImages() {
        // Given
        final int numberOfImages = mil.getNumberOfImages();
        // Then
        assertEquals(numberOfImages,NUMBER_OF_IMAGES);
    }


    @Test
    public void testGetImage() {
        byte [][]image = mil.getImage(10);

        for (byte[] bytes : image) {
            for (byte aByte : bytes) {
                System.out.print(aByte < 0 ? "0" : ".");
            }
            System.out.println();
        }

    }


}
