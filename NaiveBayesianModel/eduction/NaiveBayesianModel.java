package com.nbm;

import java.io.File;
import java.util.Scanner;
import java.util.Vector;

/**
 * Created by manshu on 2019/6/11.
 * TODO: 朴素贝叶斯分类算法
 */
public class NaiveBayesianModel {
    static Vector<String> indata = new Vector<>();//读入数据
    static Vector<int[]> catagory_R = new Vector<>();//存储类别R的所有数据
    static Vector<int[]> catagory_L = new Vector<>();//存储类别L的所有数据
    static Vector<int[]> catagory_B = new Vector<>();//存储类别B的所有数据

    public static boolean loadData(String url) {//加载测试的数据文件
        try {
            Scanner in = new Scanner(new File(url));//读入文件
            while (in.hasNextLine()) {
                String str = in.nextLine();//将文件的每一行存到str的临时变量中
                indata.add(str);//将每一个样本点的数据追加到Vector 中
            }
            return true;
        } catch (Exception e) { //如果出错返回false
            return false;
        }
    }

    public static void pretreatment(Vector<String> indata) {   //数据预处理，将原始数据中的每一个属性值提取出来存放到Vector<double[]>  data中
        int i = 0;
        String t;
        while (i < indata.size()) {//取出indata中的每一行值
            int[] tem = new int[4];
            t = indata.get(i);
            String[] sourceStrArray = t.split(",", 5);//使用字符串分割函数提取出各属性值
            switch (sourceStrArray[0]) {
                case "R": {
                    for (int j = 1; j < 5; j++) {
                        tem[j - 1] = Integer.parseInt(sourceStrArray[j]);
                    }
                    catagory_R.add(tem);
                    break;
                }
                case "L": {
                    for (int j = 1; j < 5; j++) {
                        tem[j - 1] = Integer.parseInt(sourceStrArray[j]);
                    }
                    catagory_L.add(tem);
                    break;
                }
                case "B": {
                    for (int j = 1; j < 5; j++) {
                        tem[j - 1] = Integer.parseInt(sourceStrArray[j]);
                    }
                    catagory_B.add(tem);
                    break;
                }
            }
            i++;
        }

    }
    public static double bayes(int[] x, Vector<int[]> catagory) {
        double[] ai_y = new double[4];
        int[] sum_ai = new int[4];
        for (int i = 0; i < 4; i++) {

            for (int j = 0; j < catagory.size(); j++) {
                if (x[i] == catagory.get(j)[i])
                    sum_ai[i]++;
            }
        }
        for (int i = 0; i < 4; i++) {
            ai_y[i] = (double) sum_ai[i] / (double) catagory.size();
        }
        return ai_y[0] * ai_y[1] * ai_y[2] * ai_y[3];
    }
    public static void main(String[] args) {


        loadData("D:\\work\\manshu\\src\\main\\java\\com\\nbm\\balance-scale.data");
        pretreatment(indata);
        double p_yR = (double) catagory_R.size() / (double) (indata.size());//表示概率p（R）
        double p_yB = (double) catagory_B.size() / (double) (indata.size());//表示概率p（B）
        double p_yL = (double) catagory_L.size() / (double) (indata.size());//表示概率p（L）

        int[] x = new int[4];
        double x_in_R, x_in_L, x_in_B;

        int sumR=0, sumL=0, sumB=0;
        double correct=0;


        System.out.println("请输入样本x格式如下：\n 1 1 1 1\n");
        int r = 0;
        while (r < indata.size()) {

            for (int i = 0; i < 4; i++)
                //读取数字放入数组的第i个元素
                x[i] = Integer.parseInt(indata.get(r).split(",", 5)[i + 1]);

            x_in_B = bayes(x, catagory_B) * p_yB;
            x_in_L = bayes(x, catagory_L) * p_yL;
            x_in_R = bayes(x, catagory_R) * p_yR;


            if (x_in_B == Math.max(Math.max(x_in_B, x_in_L), x_in_R)) {
                System.out.println("输入的第"+r+"样本属于类别：B");
                sumB++;
                if(indata.get(r).split(",",5)[0].equals("B"))
                    correct++;
            } else if (x_in_L == Math.max(Math.max(x_in_B, x_in_L), x_in_R)) {
                System.out.println("输入的第"+r+"样本属于类别：L");
                sumL++;
                if(indata.get(r).split(",",5)[0].equals("L"))
                    correct++;
            } else if (x_in_R == Math.max(Math.max(x_in_B, x_in_L), x_in_R)) {
                System.out.println("输入的第"+r+"样本属于类别：R");
                sumR++;
                if(indata.get(r).split(",",5)[0].equals("R"))
                    correct++;
            }



            r++;


        }


        System.out.println("使用训练样本进行分类器检验得到结果统计如下：");
        System.out.println("R类有："+sumR+"    实际有R类样本"+catagory_R.size()+"个");
        System.out.println("L类有："+sumL+"    实际有L类样本"+catagory_L.size()+"个");
        System.out.println("B类有："+sumB+"      实际有B类样本"+catagory_B.size()+"个");

        System.out.println("分类的正确率为"+correct*1.0/indata.size()*100+"%");

    }

}
