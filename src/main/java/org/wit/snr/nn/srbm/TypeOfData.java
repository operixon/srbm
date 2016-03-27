/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.nn.srbm;

import java.util.HashMap;
import java.util.Map;

/**
 * Part of minst IDX file format specyfication. <br>
 * Enum represents types off data stored in file. <br>
 * The third byte codes the type of the data. <br>
 *
 * @author koperix
 */
enum TypeOfData {

    UNSIGNED_BYTE(1, 0x08),
    SIGNED_BYTE(1, 0x09),
    SHORT(2, 0x0B),
    INT(4, 0x0C),
    FLOAT(4, 0x0D),
    DOUBLE(8, 0x0E);

    private final int numOfBytes;
    private final int code;
    
    TypeOfData(int numOfBytes, int code) {
        this.numOfBytes = numOfBytes;
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    public int getNumOfBytes() {
        return numOfBytes;
    }


}
