package org.wit.snr.minst;

import org.testng.annotations.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import static org.testng.Assert.*;

/**
 * Created by koperix on 30.03.16.
 */
public class MinstLabelLoaderTest {


    private static final int NUMBER_OF_LABELS_IN_TEST_SET = 10000;
    private static final int [] FIRST_4_LABELS_VALUE = {7,2,1,0};
    private static final int [] LAST_4_LABELS_VALUE = {3,4,5,6};


    MinstLabelLoader loader;

    @BeforeTest
    public void setUp() throws Exception {
        String f = getClass().getClassLoader().getResource("t10k-labels-idx1-ubyte").getPath();
        loader = new MinstLabelLoader(f);
        loader.loadLabels();
    }


    @Test
    public void testForFirstAndLast4LabelsLoadedFromTestFile() throws Exception {
        // Given
        final int numOffAllLoadedLabels = loader.getNumberOfLabels();
        int [] first4LoadedLab ;
        int [] last4LoadedLab ;
        // When
        first4LoadedLab = new int[]{loader.getLabel(0),loader.getLabel(1),loader.getLabel(2),loader.getLabel(3)};
        last4LoadedLab  = new int[]{
                loader.getLabel(numOffAllLoadedLabels - 4),
                loader.getLabel(numOffAllLoadedLabels - 3),
                loader.getLabel(numOffAllLoadedLabels - 2),
                loader.getLabel(numOffAllLoadedLabels - 1)};
        // Then
        assertEquals(FIRST_4_LABELS_VALUE,first4LoadedLab);
        assertEquals(LAST_4_LABELS_VALUE,last4LoadedLab);
    }

    @Test
    public void getNumberOflabelsSchouldReturnNumberOfLabelsDefinedInTestFile() throws Exception {
        // Given
        int numOffLoadedLabels ;
        // When
        numOffLoadedLabels = loader.getNumberOfLabels();
        // Then
        assertEquals(numOffLoadedLabels,NUMBER_OF_LABELS_IN_TEST_SET);
    }


}