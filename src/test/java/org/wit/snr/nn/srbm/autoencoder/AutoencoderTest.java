package org.wit.snr.nn.srbm.autoencoder;

import org.testng.annotations.Test;

import java.io.IOException;

import static org.testng.Assert.*;

public class AutoencoderTest {

    @Test
    public void testGo() throws IOException, InterruptedException {
        Autoencoder autoencoder = new Autoencoder(new CfgA(), new CfgB());
        autoencoder.go();
    }
}