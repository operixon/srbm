package org.wit.snr.nn.srbm.math.collection;

import org.testng.Assert;
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

    @Test
    void when_normalize_then_all_values_schould_be_low_than_method_param() {
        // Given
        Matrix sourceArray = Matrix2D.createFilledMatrix(10, 10, 1000);
        sourceArray.set(0, 0, 99);
        // When
        Matrix normalized = sourceArray.normalize(0, 255);

        // Then
        normalized.getDataAsList().stream().forEach(cel -> Assert.assertTrue(cel <= 255));

    }

    @Test
    void r() {
        Matrix sourceArray = Matrix2D.createFilledMatrix(10, 10, 1);
        Matrix dd = sourceArray.rowsum();
        System.out.println(sourceArray.rowsum());

    }

    @Test
    void rowSumTest() {
        // Given
        Matrix m = Matrix2D.createFilledMatrix(5, 3, 0);
        // When set first row to 1 values in each cell
        m.set(0, 0, 1);
        m.set(0, 1, 1);
        m.set(0, 2, 1);
        Matrix rowsum = m.rowsum();
        // Then
        Assert.assertEquals(rowsum.getColumnsNumber(), 1);
        Assert.assertEquals(rowsum.get(0, 0), 3.0);
        Assert.assertEquals(rowsum.get(1, 0), 0.0);
        Assert.assertEquals(rowsum.get(2, 0), 0.0);
        Assert.assertEquals(rowsum.get(3, 0), 0.0);
        Assert.assertEquals(rowsum.get(4, 0), 0.0);


    }
}