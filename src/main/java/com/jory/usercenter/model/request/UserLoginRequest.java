package com.jory.usercenter.model.request;

import lombok.Data;

import java.io.Serializable;


/**
 * @Author: Jory Zhang
 * @Date: 2024/4/27 22 26
 * @Description: 用户登录请求
 */

@Data
public class UserLoginRequest implements Serializable {

    private static final long serialVersionUID = -4459412012299784125L;

    private String userAccount;

    private String userPassword;


}
