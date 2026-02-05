package com.jory.usercenter.model.request;

import lombok.Data;

import java.io.Serializable;


/**
 * @Author: Jory Zhang
 * @Date: 2024/4/27 22 26
 * @Description: 用户注册请求
 */

@Data
public class UserRegisterRequest implements Serializable {

    private String userAccount;
    private String userPassword;
    private String checkPassword;
    private static final long serialVersionUID = -7463063316828282149L;

}
