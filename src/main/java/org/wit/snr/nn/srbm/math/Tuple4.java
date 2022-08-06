package org.wit.snr.nn.srbm.math;

public class Tuple4<A, B, C, D> {

    final private A a;
    final private B b;
    final private C c;
    final private D d;

    public Tuple4(A a, B b, C c, D d) {
        this.a = a;
        this.b = b;
        this.c = c;
        this.d = d;
    }

    public A getA() {
        return a;
    }

    public B getB() {
        return b;
    }

    public C getC() {
        return c;
    }

    public D getD() {
        return d;
    }
}
