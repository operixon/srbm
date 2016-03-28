/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.minst;

/**
 *
 * @author koperix
 */
public class IdxFileMetadata {

    private final TypeOfData typeOfData;
    private final int numberOfDimensions;
    private final int[] sizeInDimension;

    public IdxFileMetadata(TypeOfData typeOfData, int numberOfDimensions, int[] sizeInDimension) {
        this.typeOfData = typeOfData;
        this.numberOfDimensions = numberOfDimensions;
        this.sizeInDimension = sizeInDimension;
    }

    public TypeOfData getTypeOfData() {
        return typeOfData;
    }

    public int getNumberOfDimensions() {
        return numberOfDimensions;
    }

    public int[] getSizeInDimension() {
        return sizeInDimension;
    }

    
    
    
    
    
    

}
