package com.kmeans;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by manshu on 2019/8/8.
 * TODO:聚类算法 落地 平均值k
 */
public class Kmeans {
    private int numOfClass;// 分类
    private int iteration;// 迭代次数
    private int dataSetLength;//数据集的长度
    private ArrayList<float[]> dataSet;// 数据集链表
    private ArrayList<float[]> center;// 中心链表
    private ArrayList<ArrayList<float[]>> cluster; //类别
    private ArrayList<Float> sumOfError;// 误差平方和
    private Random random;
    //初始化每一类的质心 中心点集
    private ArrayList<float[]> initCenters() {
        ArrayList<float[]> center = new ArrayList<float[]>();
        int[] randoms = new int[numOfClass];
        boolean flag;
        int temp = random.nextInt(dataSetLength);
        randoms[0] = temp;
        //randoms数组中存放数据集的不同的下标
        for (int i = 1; i < numOfClass; i++) {
            flag = true;
            while (flag) {
                temp = random.nextInt(dataSetLength);

                int j=0;
                for(j=0; j<i; j++){
                    if(randoms[j] == temp){
                        break;
                    }
                }

                if (j == i) {
                    flag = false;
                }
            }
            randoms[i] = temp;
        }

        // 测试随机数生成情况
        // for(int i=0;i<numOfClass;i++)
        // {
        // System.out.println("test1:randoms["+i+"]="+randoms[i]);
        // }

        // System.out.println();

        for (int i = 0; i < numOfClass; i++) {
            center.add(dataSet.get(randoms[i]));// 生成初始化中心链表
        }
        return center;
    }

    //初始化类集合 空数据的类集合
    private ArrayList<ArrayList<float[]>> initCluster() {
        ArrayList<ArrayList<float[]>> cluster = new ArrayList<ArrayList<float[]>>();
        for (int i = 0; i < numOfClass; i++) {
            cluster.add(new ArrayList<float[]>());
        }

        return cluster;
    }

    //计算两个点之间的距离
    private float distance(float[] element, float[] center) {
        float distance = 0.0f;
        float x = element[0] - center[0];
        float y = element[1] - center[1];
        float z = x * x + y * y;
        distance = (float) Math.sqrt(z);

        return distance;
    }

    // 获取距离集合中最小距离的位置
    private int minDistance(float[] distance) {
        float minDistance = distance[0];
        int minLocation = 0;
        for (int i = 1; i < distance.length; i++) {
            if (distance[i] <= minDistance) {
                minDistance = distance[i];
                minLocation = i;
            }
        }
        return minLocation;
    }

    // 核心，将当前元素放到最小距离中心相关的簇中
    private void clusterSet() {
        float[] distance = new float[numOfClass];
        for (int i = 0; i < dataSetLength; i++) {
            for (int j = 0; j < numOfClass; j++) {
                distance[j] = distance(dataSet.get(i), center.get(j));
                // System.out.println("test2:"+"dataSet["+i+"],center["+j+"],distance="+distance[j]);
            }
            int minLocation = minDistance(distance);
            // System.out.println("test3:"+"dataSet["+i+"],minLocation="+minLocation);
            // System.out.println();

            cluster.get(minLocation).add(dataSet.get(i));// 核心，将当前元素放到最小距离中心相关的簇中

        }
    }

     // 求两点误差平方的方法
    private float errorSquare(float[] element, float[] center) {
        float x = element[0] - center[0];
        float y = element[1] - center[1];

        float errSquare = x * x + y * y;

        return errSquare;
    }

    //计算误差平方和准则函数方法
    private void countRule() {
        float jcF = 0;
        for (int i = 0; i < cluster.size(); i++) {
            for (int j = 0; j < cluster.get(i).size(); j++) {
                jcF += errorSquare(cluster.get(i).get(j), center.get(i));

            }
        }
        sumOfError.add(jcF);
    }

    //设置新的簇中心方法
    private void setNewCenter() {
        for (int i = 0; i < numOfClass; i++) {
            int n = cluster.get(i).size();
            if (n != 0) {
                float[] newCenter = { 0, 0 };
                for (int j = 0; j < n; j++) {
                    newCenter[0] += cluster.get(i).get(j)[0];
                    newCenter[1] += cluster.get(i).get(j)[1];
                }
                // 设置一个平均值
                newCenter[0] = newCenter[0] / n;
                newCenter[1] = newCenter[1] / n;
                center.set(i, newCenter);
            }
        }
    }

    //Kmeans算法核心过程方法
    private void kmeans() {
        init();
        // printDataArray(dataSet,"initDataSet");
        // printDataArray(center,"initCenter");

        // 循环分组，直到误差不变为止
        while (true) {
            clusterSet();
            // for(int i=0;i<cluster.size();i++)
            // {
            // printDataArray(cluster.get(i),"cluster["+i+"]");
            // }

            countRule();

            // System.out.println("count:"+"sumOfError["+iteration+"]="+sumOfError.get(iteration));

            // System.out.println();
            // 误差不变了，分组完成
            if (iteration != 0) {
                if (sumOfError.get(iteration) - sumOfError.get(iteration - 1) == 0) {
                    break;
                }
            }

            setNewCenter();
            // printDataArray(center,"newCenter");
            iteration++;
            cluster.clear();
            cluster = initCluster();
        }

        // System.out.println("note:the times of repeat:iteration="+iteration);//输出迭代次数
    }
    //执行算法
    public void execute() {
        long startTime = System.currentTimeMillis();
        System.out.println("kmeans begins");
        kmeans();
        long endTime = System.currentTimeMillis();
        System.out.println("kmeans running time=" + (endTime - startTime)
                + "ms");
        System.out.println("kmeans ends");
        System.out.println();
    }
    //初始化数据集
    public void setDataSet(ArrayList<float[]> dataSet) {
        this.dataSet = dataSet;
    }
    // 获取分类集合
    public ArrayList<ArrayList<float[]>> getCluster() {
        return cluster;
    }
    //构造函数，传入自定义k 若numOfClass<=0时，设置为1，若numOfClass大于数据源的长度时，置为数据源的长度 分类
    public Kmeans(int numOfClass) {
        if (numOfClass <= 0) {
            numOfClass = 1;
        }
        this.numOfClass = numOfClass;
    }
    //如果调用者未初始化数据集，则采用内部测试数据集
    private void init() {
        iteration = 0;
        random = new Random();
        //如果调用者未初始化数据集，则采用内部测试数据集
        if (dataSet == null || dataSet.size() == 0) {
            initDataSet();
        }
        dataSetLength = dataSet.size();
        //若numOfClass大于数据源的长度时，置为数据源的长度
        if (numOfClass > dataSetLength) {
            numOfClass = dataSetLength;
        }
        center = initCenters();
        cluster = initCluster();
        sumOfError = new ArrayList<Float>();
    }
    //如果调用者未初始化数据集，则采用内部测试数据集
    private void initDataSet() {
        dataSet = new ArrayList<float[]>();
        // 其中{6,3}是一样的，所以长度为15的数据集分成14簇和15簇的误差都为0
        float[][] dataSetArray = new float[][] { { 8, 2 }, { 3, 4 }, { 2, 5 },
                { 4, 2 }, { 7, 3 }, { 6, 2 }, { 4, 7 }, { 6, 3 }, { 5, 3 },
                { 6, 3 }, { 6, 9 }, { 1, 6 }, { 3, 9 }, { 4, 1 }, { 8, 6 } };

        for (int i = 0; i < dataSetArray.length; i++) {
            dataSet.add(dataSetArray[i]);
        }
    }
    //打印数据，测试用
    public void printDataArray(ArrayList<float[]> dataArray,
                               String dataArrayName) {
        for (int i = 0; i < dataArray.size(); i++) {
            System.out.println("print:" + dataArrayName + "[" + i + "]={"
                    + dataArray.get(i)[0] + "," + dataArray.get(i)[1] + "}");
        }
        System.out.println("===================================");
    }

}
