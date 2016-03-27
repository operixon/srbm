/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

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
public class MinstUtilsNGTest {
    
    public MinstUtilsNGTest() {
    }

    @BeforeClass
    public static void setUpClass() throws Exception {
    }

    @AfterClass
    public static void tearDownClass() throws Exception {
    }

    @BeforeMethod
    public void setUpMethod() throws Exception {
    }

    @AfterMethod
    public void tearDownMethod() throws Exception {
    }

    /**
     * Test of getTypeOfDataByCode method, of class MinstUtils.
     */
    @Test
    public void testGetTypeOfDataByCode() {
        
        for(TypeOfData tod : TypeOfData.values()){
            // When
            TypeOfData typeOfDataByCode = MinstUtils.getTypeOfDataByCode(tod.getCode());
            //Then
            assertEquals(typeOfDataByCode, tod);
        }
    }

    
}
