package org.wit.snr.nn.srbm;

import org.wit.snr.nn.srbm.math.collection.Matrix;

import java.util.Optional;

public class MiniBatchTrainingResult
{

    final Matrix W;
    final Matrix vbias;
    final Matrix hbias;

    public MiniBatchTrainingResult(Matrix w, Matrix vbias, Matrix hbias)
    {
        W = w;
        this.vbias = vbias;
        this.hbias = hbias;
    }

    public Matrix getW()
    {
        return W;
    }

    public Matrix getVbias()
    {
        return vbias;
    }

    public Matrix getHbias()
    {
        return hbias;
    }

    public Optional<MiniBatchTrainingResult> apply(Optional<MiniBatchTrainingResult> p2)
    {
        return Optional.of(
                new MiniBatchTrainingResult(
                        getW().matrixAdd(p2.get().getW()),
                        getVbias().matrixAdd(p2.get().getVbias()),
                        getHbias().matrixAdd(p2.get().getHbias())
                ));
    }
}
