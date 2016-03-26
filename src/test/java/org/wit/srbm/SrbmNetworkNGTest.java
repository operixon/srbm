/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.srbm;

import org.wit.snr.nn.srbm.LearningAlgorithm;
import static org.testng.Assert.*;

/**
 *
 * @author koperix
 */
public class SrbmNetworkNGTest {
    
    public SrbmNetworkNGTest() {
    }

    @org.testng.annotations.BeforeClass
    public static void setUpClass() throws Exception {
    }

    @org.testng.annotations.AfterClass
    public static void tearDownClass() throws Exception {
    }

    @org.testng.annotations.BeforeMethod
    public void setUpMethod() throws Exception {
    }

    @org.testng.annotations.AfterMethod
    public void tearDownMethod() throws Exception {
    }

    /**
     * Test of main method, of class LearningAlgorithm.
     */
    @org.testng.annotations.Test
    public void testLearning() {
        
        LearningAlgorithm algorithm = new LearningAlgorithm();
        algorithm.train();
        
        
        
    }
    
}
