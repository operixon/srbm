package org.wit.snr.nn.srbm.math.collection;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class MatrixTest {

    @BeforeMethod
    public void setUp() {
    }

    @AfterMethod
    public void tearDown() {
    }

    @Test
    public void testCreateMatrixWithRandomValues() {
        Matrix m = Matrix.createMatrixWithRandomValues(2, 10);
        Assert.assertEquals(m.getRows(), 2);
        Assert.assertEquals(m.getColumns(), 10);
        m.getDataAsList().stream().forEach(val -> Assert.assertTrue(val >= 0));
        m.getDataAsList().stream().forEach(val -> Assert.assertTrue(val <= 1));
    }

    @Test
    public void testCreateMatrixFromArray() {
    }

    @Test
    public void testToFullString() {
    }

    @Test
    public void testToString() {
    }

    @Test
    public void testGet() {
    }

    @Test
    public void testSet() {
    }

    @Test
    public void testSubstract() {
    }

    @Test
    public void testScalarDivide() {
    }

    @Test
    public void testScalarMultiply() {
    }

    @Test
    public void testMatrixAdd() {
    }

    @Test
    public void testGetDataAsList() {
    }

    @Test
    public void testTranspose() {
    }

    @Test
    public void testMultiplication() {
    }
}