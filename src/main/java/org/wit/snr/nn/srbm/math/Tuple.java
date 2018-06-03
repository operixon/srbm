package org.wit.snr.nn.srbm.math;

public class Tuple<A, B> {

    final private A a;
    final private B b;

    public Tuple(A a, B b) {
        this.a = a;
        this.b = b;
    }

    public A getA() {
        return a;
    }

    public B getB() {
        return b;
    }
}
