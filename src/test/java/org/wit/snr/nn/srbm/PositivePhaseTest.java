package org.wit.snr.nn.srbm;

import org.junit.BeforeClass;
import org.testng.annotations.Test;
import org.wit.snr.nn.srbm.layer.Equation3;
import org.wit.snr.nn.srbm.layer.Layer;
import org.wit.snr.nn.srbm.layer.PositivePhaseComputations;
import org.wit.snr.nn.srbm.math.ActivationFunction;
import org.wit.snr.nn.srbm.math.collection.Matrix;
import org.wit.snr.nn.srbm.math.collection.Matrix2D;
import org.wit.snr.nn.srbm.math.function.SigmoidFunction;

import java.util.List;

public class PositivePhaseTest {

    private static Configuration cfg = new CfgForPositivePhase();


    @Test
    void testEquation3(){
        // Given
        ActivationFunction activationFunction = new SigmoidFunction();
        Layer layer = new Layer(cfg.numdims(),cfg.numhid());
        Equation3 eq = new Equation3(cfg, layer, activationFunction);
        List<Double> sample = Matrix2D.createFilledMatrix(1, 0, 1).getColumn(0);
        Double value = eq.evaluate(0, sample, 0.5);
    }

    @Test
    void t1()
    {
        //PositivePhaseComputations computations = new PositivePhaseComputations(new Configuration(), )

    }

}
