package org.wit.snr.nn.srbm;

import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

public class RBMTest {

    @Test
    public void testTrain() {

        List<Double> l = new ArrayList<>();
        l.add(2.4);
        l.add(1.3);
        l.forEach(x -> x = x + 2);
        System.out.println(l);
    }
}