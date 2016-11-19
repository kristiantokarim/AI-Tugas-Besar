package aitubesweka;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

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
    
    public void generateFreqTable(Instances ins) {
        probTable = new double[classCount][attrCount][0];
        freqTable = new int[classCount][attrCount][0];
        for (int i = 0 ; i < attrCount ; i++) {
            Attribute proAttr = ins.attribute(i);
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount; j++) {
                    probTable[j][i] = new double[proAttr.numValues()];
                    freqTable[j][i] = new int[proAttr.numValues()];
                }
            }
            else {
                for (int j = 0 ; j < classCount; j++) {
                    probTable[j][i] = new double[1];
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
                    freqTable[(int)proIns.classValue()][j][(int)proIns.value(proAttr)]++;
                }
            }
            classFreqTable[(int)proIns.classValue()]++;
        }
    }
    
    public void generateProbTable (Instances ins) {
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
    public void buildClassifier(Instances ins) throws Exception {
        
        getCapabilities().testWithFail(ins);

        Discretize discretizeFilter = new Discretize();
        discretizeFilter.setInputFormat(ins);
        ins = Filter.useFilter(ins, discretizeFilter);
        System.out.println(ins.attribute(0).numValues());
        proInstances = new Instances(ins,0);
        classCount = ins.numClasses();
        attrCount = ins.numAttributes() - 1;
        generateFreqTable(ins);
        generateProbTable(ins);
    }
    
    @Override
    public double[] distributionForInstance(Instance ins) {
        double[] res = new double[classCount];
        double result = 1;
        for (int i =0;i<classCount;i++){
            for (int j=0; j<attrCount;j++){
                String val = ins.stringValue(j);
                
                int valIdx = proInstances.attribute(i).indexOfValue(val);
                result *= probTable[i][valIdx][0]; 
            }
            res[i] = classProbTable[i]*result;
        }
        // P = P(kelas)*P(att1|kelas)*P(att2*|kelas)....
        return res;
    }
}
