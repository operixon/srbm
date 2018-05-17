package org.wit.snr.nn.srbm.math.collection;

import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class Matrix2DTest {

    @BeforeMethod
    public void setUp() {
    }

    @AfterMethod
    public void tearDown() {
    }

    @Test
    public void testCreateFilledMatrix() {
        Matrix m = Matrix2D.createFilledMatrix(1, 100, 1.0);
    }
}