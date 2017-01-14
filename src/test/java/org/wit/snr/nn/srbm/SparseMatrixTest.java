package org.wit.snr.nn.srbm;

import org.testng.annotations.Test;

import static org.testng.Assert.*;

/**
 * Created by kkoperkiewicz on 13.01.2017.
 */
public class SparseMatrixTest {

    @Test
    public void testXYPairUniq() throws Exception {

        // Given
        int x = 10;
        int y = 12;
        int value1 = 5;
        int value2 = 14;
        SparseMatrix<Integer> m = new SparseMatrix<>(20,20);

        //When
        m.set(x,y,value1);
        m.set(y,x,value2);

        //then
        assertNotEquals(m.get(x,y),m.get(y,x));
    }

    @Test
    public void schould_throw_exception_on_index_overflow() throws Exception {
        //Given
        SparseMatrix<Integer> m = new SparseMatrix<>(10,10);
        //when
        Exception ioutException = null;
        try{
            m.set(12,12,500);
        } catch (IndexOutOfBoundsException e) {
            ioutException = e;
        }
        //then
        assertNotNull(ioutException);
    }

    @Test
    public void test_get_sparse_factor(){
        // For given 3x2 empty matrix
        SparseMatrix<Integer> m = new SparseMatrix<>(3,2);
        // when  using half matrix capacity
        for(int x =0; x < 3; x++){
            m.set(x,0,1);
        }
        // then sparsnes factor is 50%
        assertEquals(m.getSparsnesFactor(),50.0);
    }

}