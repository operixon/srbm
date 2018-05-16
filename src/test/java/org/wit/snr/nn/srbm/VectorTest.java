package org.wit.snr.nn.srbm;

import org.testng.annotations.Test;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Vector;

import java.util.Arrays;

/**
 * Created by kkoperkiewicz on 11.06.2017.
 */
public class VectorTest {
    
    
    @Test
    public void test1(){
        Vector v1 = new Vector(Arrays.asList(new Double[] {1.0,1.0}));
        Vector v2 = new Vector(Arrays.asList(new Double[] {0.0,0.0,1.0,1.0}));
        Matrix matrix = v1.multiplyByTransposedVector(v2);
        System.out.println(matrix.toString());

    }

}