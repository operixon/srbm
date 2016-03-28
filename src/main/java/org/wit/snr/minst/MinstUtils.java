/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.minst;

import org.wit.snr.minst.TypeOfData;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author koperix
 */
public class MinstUtils {

    private static final Map<Integer, TypeOfData> codeToEnumCache = new HashMap<>();

    private static void updateCodeToEnumCache() {
        if (codeToEnumCache.isEmpty()) {
            for (TypeOfData t : TypeOfData.values()) {
                codeToEnumCache.put(t.getCode(), t);
            }
        }
    }

    static TypeOfData getTypeOfDataByCode(int code) {
        if (codeToEnumCache.isEmpty()) {
            updateCodeToEnumCache();
        }
        if (codeToEnumCache.containsKey(code)) {
            return codeToEnumCache.get(code);
        } else {
            throw new IllegalArgumentException(String.format("Ni ma takiego kodu %02x. A to bardzo niedobrze.", code));
        }
    }

}
