package com.jory.usercenter.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.jory.usercenter.model.User;

import javax.servlet.http.HttpServletRequest;

/**
* @author Jory Zhang
* @description 针对表【user】的数据库操作Service
* @createDate 2024-04-27 10:00:07
*/

/**
 *  用户服务
 */
public interface UserService extends IService<User> {

    /**
     * 用户注册
     * @param userAccount 用户账户
     * @param userPassword 用户密码
     * @param checkPassword 校验密码
     * @return 新用户id
     */

    long userRegister(String userAccount, String userPassword, String checkPassword);

    /**
     * 用户登录
     * @param userAccount
     * @param userPassword
     * @param request
     * @return
     */
    User userLogin(String userAccount, String userPassword, HttpServletRequest request);

    /**
     * 用户脱敏
     * @param originUser
     * @return
     */
    User getSafeUser(User originUser);

    /**
     * 用户注销
     * @param request
     * @return
     */
    int userOutLogin(HttpServletRequest request);
}
