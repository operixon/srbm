/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wit.snr.minst;

import java.io.IOException;

/**
 *
 * @author koperix
 */
public class MinstLabelLoader  extends IdxFileInMemory{

    
    
    public MinstLabelLoader(String imagesPath) throws IOException {
        super(imagesPath);
    }

    
    public void loadLabels() throws IOException{
        loadData();
    }
    
    public int getLabel(int labelIdx){
        return data[labelIdx];
    }
    
    public int getNumberOfLabels(){
        return fileMetadata.getSizeInDimension()[0];
    }
    
    public byte[] getLabels(){
        return data;
    }
    

}
