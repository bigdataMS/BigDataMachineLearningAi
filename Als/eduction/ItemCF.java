package com.cf;

import java.io.*;
import java.util.*;

/**
 * Created by manshu on 2018/6/24.
 * TODO: 协同过滤
 * 假设 –> 用户喜欢跟他过去喜欢的物品相似的物品 –> 历史上相似的物品在未来也相似 -----> 给定用户u，找到他过去喜欢的物品的集合R(u). –> 把和R(u)相似的物品推荐给u.
 */
public class ItemCF {
    //存储数据内容
    private static List<String[]> list=new ArrayList<String[]>();
    //Item聚合
    private static String item="";
    //归一化中间集合
    private static List<String[]> mid_list=new ArrayList<String[]>();
    //归一化后数据
    private static List<String[]> nm_list=new ArrayList<String[]>();
    //Uer聚合
    private static String user="";;
    //衍生Pair对->分数中间集合
    private static List<String[]> pmid_list=new ArrayList<String[]>();
    //衍生Pair对->分数
    private static List<String[]> pair_list=new ArrayList<String[]>();
    //pair聚合
    private static String pair="";
    //聚合分数
    private static Double score=0.0;
    //结果集
    private static List<String[]> result_list=new ArrayList<String[]>();
    //归一化
    public static void normalization(){
        list.forEach((String[] arr)->{
            if(item.equals("")){
                item=arr[1];
            }
            if(!item.equals(arr[1])){
                Double sum=0.0;
                for(int i=0;i<mid_list.size();i++){
                    sum+=Math.pow(Double.parseDouble(mid_list.get(i)[1]),2);
                }
                for(String[] str:mid_list){
                    nm_list.add(new String[]{str[0],arr[1],String.valueOf(Double.parseDouble(str[1])/sum)});
                }
                mid_list.clear();
                item=arr[1];
            }
            mid_list.add(new String[]{arr[0],arr[1]});
        });
        Double sum=0.0;
        for(int i=0;i<mid_list.size();i++){
            sum+=Math.pow(Double.parseDouble(mid_list.get(i)[1]),2);
        }
        for(String[] str:mid_list){
            nm_list.add(new String[]{str[0],item,String.valueOf(Double.parseDouble(str[1])/sum)});
        }
    }
    //按照user聚合
    public static void userpoly(){
        nm_list.forEach((String[] arr)->{
            if(user.equals("")){
                user=arr[0];
            }
            if(!user.equals(arr[0])){
                for(int i=0;i<pmid_list.size()-1;i++){
                    pair_list.add(new String[]{
                            pmid_list.get(i)[0]+","+pmid_list.get(i+1)[0],
                            String.valueOf(Double.parseDouble(pmid_list.get(i)[1])*Double.parseDouble(pmid_list.get(i+1)[1]))});
                    pmid_list.clear();
                    user=arr[0];
                }
            }
            pmid_list.add(new String[]{arr[0],arr[1]});
        });
        for(int i=0;i<pmid_list.size()-1;i++){
            pair_list.add(new String[]{
                    pmid_list.get(i)[0]+","+pmid_list.get(i+1)[0],
                    String.valueOf(Double.parseDouble(pmid_list.get(i)[1])*Double.parseDouble(pmid_list.get(i+1)[1]))});
        }
    }
    //按照pair聚合
    public static void pairpoly(){
        pair_list.forEach((String[] arr)->{
            if(!pair.equals("")){
                pair=arr[0];
            }
            if(!pair.equals(arr[0])){
                result_list.add(new String[]{arr[0],String.valueOf(score)});
                pair=arr[0];
                score=0.0;
            }
            score+=Double.parseDouble(arr[1]);
        });
        result_list.add(new String[]{pair,String.valueOf(score)});
    }
    //加载数据
    public static void load(){
        try {
            //File file = new File("D:\\algorithm\\mr_cf\\music_uis.data");
            File file = new File("D:\\algorithm\\mr_cf\\part-00151");
            InputStreamReader input = new InputStreamReader(new FileInputStream(file));
            BufferedReader br = new BufferedReader(input);
            String data;
            while ((data = br.readLine()) != null){

                //String[] result=data.split("\t");
                String[] result=data.split(",");
                if(result.length!=3){
                    continue;
                }else{
                    list.add(result);
                }
            }
            br.close();
            input.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    public static void main(String[] args) {

        load();
        normalization();
        userpoly();
        userpoly();
        pairpoly();
        System.out.println(list.get(0)[0]+"----"+list.get(0)[1]+"----"+list.get(0)[2]);
    }

}
