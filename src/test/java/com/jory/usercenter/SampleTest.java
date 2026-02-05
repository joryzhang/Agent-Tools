package com.jory.usercenter;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import javax.annotation.Resource;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


@SpringBootTest
public class SampleTest {

    //Autowired是按照类型自动注入
    //Resource是按照Java Bean的名称去注入


    @Test
    public void testSelect() {

    }

    public static void main(String[] args) {


        List<String> strList = new ArrayList<>();
        strList.add("jory");
        strList.add("jory");
        strList.add("jory");
        strList.add("tsfsfsdfs");
        Map<String, Integer> stringIntegerMap = calculateStringCount(strList);
        for (Map.Entry<String, Integer> stringIntegerEntry : stringIntegerMap.entrySet()) {
            System.out.println(stringIntegerEntry);
        }

    }
    public static Map<String, Integer> calculateStringCount(List<String> strList) {
        Map<String,Integer> map = new HashMap<>();
        for(String str: strList){
            map.get(str);
            map.put(str,map.get(str)+1);
        }
        return map;
    }
}


