package org.wit.snr.nn.srbm;

import org.testng.annotations.Test;
import org.wit.snr.nn.srbm.math.function.SigmoidFunction;

public class SigmoidFunctionTest {

    @Test
    public void testEvaluate() {

        SigmoidFunction fun = new SigmoidFunction();
        System.out.println(fun.evaluate(0));
        System.out.println(fun.evaluate(1));
        System.out.println(fun.evaluate(-1));
        System.out.println(fun.evaluate(-1000));
        System.out.println(fun.evaluate(10));
        System.out.println(fun.evaluate(4));
        System.out.println(fun.evaluate(300));


    }
}