package aitubesweka;


import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author programming
 */
public class naiveBayes extends AbstractClassifier {
    public Instances proInstances;
    public int classCount;
    public int attrCount;
    public int [][][] freqTable; 
    public int [] classFreqTable;
    public double [][][] probTable;
    public double [] classProbTable;
    
    /**
     *
     * @param ins
     * @throws Exception
     */
    public void buildClassifier(Instances ins) throws Exception {
        
        if (ins.checkForStringAttributes()) {
            throw new Exception("Can't handle string attributes!");
        }
        if (ins.classAttribute().isNumeric()) {
            throw new Exception("Naive Bayes: Class is numeric!");
        }
        
        proInstances = new Instances(ins,0);
        classCount = ins.numClasses();
        attrCount = ins.numAttributes() - 1;
        
        probTable = new double[classCount][attrCount][0];
        for (int i = 0 ; i < attrCount ; i++) {
            Attribute proAttr = ins.attribute(i);
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount; j++) {
                    probTable[j][i] = new double[proAttr.numValues()];
                }
            }
            else {
                for (int j = 0 ; j < classCount; j++) {
                    probTable[j][i] = new double[1];
                }
            }
        }
        
        freqTable = new int[classCount][attrCount][0];
        for (int i = 0 ; i < attrCount ; i++) {
            Attribute proAttr = ins.attribute(i);
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount; j++) {
                    freqTable[j][i] = new int[proAttr.numValues()];
                }
            }
            else {
                for (int j = 0 ; j < classCount; j++) {
                    freqTable[j][i] = new int[1];
                }
            }
        }
        
        classFreqTable = new int[classCount];
        classProbTable = new double[classCount];
        for (int i = 0 ; i < ins.numInstances() ; i++) {
            Instance proIns = ins.get(i);
            if (!proIns.classIsMissing()) {
                for (int j = 0 ; j < attrCount ; j++) {
                    Attribute proAttr = ins.attribute(j);
                    if (proAttr.isNominal()) {
                        freqTable[(int)proIns.classValue()][j][(int)proIns.value(proAttr)]++;
                    }
                    else {
                        freqTable[(int)proIns.classValue()][j][0]++;
                    }
                }
            }
            classFreqTable[(int)proIns.classValue()]++;
        }
        
        for (int i = 0 ; i < attrCount ; i++) {
            Attribute proAttr = ins.attribute(i);
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount ; j++) {
                    int sum = Utils.sum(freqTable[j][i]);
                    for (int k = 0 ; k < proAttr.numValues() ; k++) {
                        probTable[j][i][k] = ((double)freqTable[j][i][k])/((double)sum);
                    }
                }
            }
        } 
        
        int sum = Utils.sum(classFreqTable);
        for (int i = 0 ; i < classCount ; i++) {
            classProbTable[i] = ((double)classFreqTable[i])/((double)sum);
        }
    }
    
}
