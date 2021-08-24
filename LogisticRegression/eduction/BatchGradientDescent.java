package com.lr;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by manshu on 2018/3/18.
 * TODO: 逻辑回归 批量梯度下降（Batch Gradient Descent）BGD
 */
public class BatchGradientDescent {
    //存储数据内容
    private static List<Double[]> list=new ArrayList<Double[]>();
    //构建训练集矩阵
    private static Double[][] Matrix;
    // 步长->学习率
    private static  double alpha = 0.001;
    // 迭代次数
    private static  int steps = 500;
    //初始化权重向量
    private static Double[][] weights;
    //初始化分类标签列表
    private static Double[][] target;
    //构建训练集矩阵
    public static void geMatrix(){
            //开始构建x+b 系数矩阵：b这里默认为1
            //初始化第一列默认1
            Matrix= new Double[list.size()][list.get(0).length];
            for(int i=0;i<list.size();i++){
                Matrix[i][0]=1.0;
            }
            //初始化第二列->值为list.get(i)数组中的第一列
            for(int i=0;i<list.size();i++){
                Matrix[i][1]=list.get(i)[0];
            }
            //初始化第二列->值为list.get(i)数组中的第二列
            for(int i=0;i<list.size();i++){
                Matrix[i][2]=list.get(i)[1];
            }
            //训练集矩阵构建完成list.size()个样本，特征list.get(i).length  矩阵(list.size()维度,list.get(i).length维度)


    }
    //初始化权重向量矩阵和真实标签矩阵
    public static void initWeights(){
        weights=new Double[list.get(0).length][1];
        weights[0][0]=1.0;
        weights[1][0]=1.0;
        weights[2][0]=1.0;
        target=new Double[list.size()][1];
        for(int i=0;i<list.size();i++){
            target[i][0]=list.get(i)[2];
        }

    }
    // Logistic函数->sigmoid
    public static Double[][] sigmoid(Double[][] wx) {
        Double[][] sigmod=new Double[wx.length][wx[0].length];
        for(int i=0;i<wx.length;i++){
            double v = 1.0 / (1 + Math.exp(-wx[i][0]));
            sigmod [i][0]=v;
        }
        return sigmod;
    }
    //矩阵相乘
    public static Double[][] MatrixMutMatrix(Double a[][], Double b[][]) {
        int arow = a.length;
        int bcol = b[0].length;
        int m = b.length;
        Double[][] c = new Double[arow][bcol];
        for (int i = 0; i < arow; i++) {
            for (int j = 0; j < bcol; j++) {
                Double result = 0.0;
                for (int k = 0; k < m; k++) {
                    result += a[i][k] * b[k][j];
                }
                c[i][j] = result;
            }
        }
        return c;
    }
    //矩阵相减->计算误差
    public static Double[][] subMatrix(Double[][] A, Double[][] B){
        int line=A.length,list=A[0].length;
        Double[][] C =new Double[line][list];
        for(int i=0;i<line;i++)
        {
            for(int j=0;j<list;j++)
            {
                C[i][j]=A[i][j]-B[i][j];
            }

        }
        return C;
    }
    // 将矩阵转置
    public static Double[][] revMatrix(Double temp [][]) {
        Double[][] result =new Double[temp[0].length][temp.length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = temp[j][i] ;

            }
        }
        return result;
    }
    // 将矩阵乘以一个数
    public static Double[][] mutMatrix(Double temp [][],Double v) {
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                temp[i][j] = temp[i][j]*v;
            }
        }
        return temp;
    }
    //矩阵相加
    public static Double[][] AddsMatrix(Double[][]A,Double[][] B){
        int line=A.length,list=A[0].length;
        Double[][]C=new Double[line][list];
        for(int i=0;i<line;i++)
        {
            for(int j=0;j<list;j++)
            {
                C[i][j]=A[i][j]+B[i][j];
            }

        }
        return C;
    }
    //回归函数
    public static Double regression_calc(Double[][] w,Double[][] x){
        Double[][] result=sigmoid(MatrixMutMatrix(w,x));
        Double value=result[0][0];
        return value;
    }
    //分类函数
    public static Double classifier(Double[][]x,Double[][] w){
        Double[][] result=sigmoid(MatrixMutMatrix(w,x));
        Double value=result[0][0];
        Double v;
        if(value>0.5){
            v=1.0;
        }else{
            v=0.0;
        }
        return v;
    }
    //解析数据
    public static void getDate(){
        try {
            File file = new File("D:\\data\\lr_data\\testSet.txt");
            InputStreamReader inputReader = new InputStreamReader(new FileInputStream(file));
            BufferedReader bf = new BufferedReader(inputReader);
            String str;
            while ((str = bf.readLine()) != null){
                Double[] arr=new Double[3];
                String[] result=str.split("\t");
                for(int i=0;i<result.length;i++){
                    arr[i]=Double.parseDouble(result[i]);
                }
                list.add(arr);
            }
            bf.close();
            inputReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    /**
     *
     * @param args
     * 1、设置初始w，计算F(w)
     * 2、计算梯度 • 下降方向
     * 3、尝试梯度更新
     * 4、如果 较小，停止； 否则 ；跳到第2步
     */
    public static void main(String[] args) {
        getDate();
        geMatrix();
        initWeights();
        for(int i=0;i<steps;i++){
            //训练集矩阵 乘  权重  w*x
            Double[][] gradient=MatrixMutMatrix(Matrix,weights);
            //sigmoid函数  1/1+exp(-wx) 返回预测值
            Double[][] output=sigmoid(gradient);
            //真实值减预测值  返回误差
            Double[][] errors = subMatrix(target,output);
            //训练集矩阵 转置
            Double[][] dataMat=revMatrix(Matrix);
            //转置后的训练集矩阵 乘 步长
            Double[][] mut=mutMatrix(dataMat,alpha);
            //所有样本乘以误差
            Double[][] err=MatrixMutMatrix(mut,errors);
            //更新权重  权重 +步长∗ 梯度（误差）
            weights = AddsMatrix(weights,err);
        }
        System.out.println(weights[0][0]);
        System.out.println(weights[1][0]);
        System.out.println(weights[2][0]);
        /*得到权重
        4.178813076565532
        0.5048987439366058
        0.6198026439379993*/
        Double[][] x=new Double[1][3];
        x[0][0]=1.0;
        x[0][1]=0.9316350;
        x[0][2]=-1.589505;
        Double[][] w=new Double[1][3];
        w[0][0]=4.178813076565532;
        w[0][1]=0.5048987439366058;
        w[0][2]=0.6198026439379993;
        //回归函数
        Double a=regression_calc(w,x);
        //分类函数
        Double b=classifier(w,x);
        System.out.println(a);
        System.out.println(b);
    }
}
