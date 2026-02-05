package com.jory.usercenter.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.jory.usercenter.common.BaseResponse;
import com.jory.usercenter.common.ErrorCode;
import com.jory.usercenter.common.ResultUtils;
import com.jory.usercenter.exception.BusinessException;
import com.jory.usercenter.model.User;
import com.jory.usercenter.service.UserService;
import com.jory.usercenter.mapper.UserMapper;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomStringUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.util.DigestUtils;

import javax.servlet.http.HttpServletRequest;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.jory.usercenter.constant.UserConstant.USER_LOGIN_STATE;

/**
 * @author Jory Zhang
 * @description 针对表【user】的数据库操作Service实现
 * @createDate 2024-04-27 10:00:07
 */
@Service
@Slf4j
public class UserServiceImpl extends ServiceImpl<UserMapper, User>
        implements UserService {

    /**
     * 盐值
     */
    private static final String SALT = "jory";


    @Override
    public long userRegister(String userAccount, String userPassword, String checkPassword) {
        //1. 校验
        if (StringUtils.isAnyBlank(userAccount, userPassword, checkPassword)) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"参数为空");
        }
        //2.校验账户
        if (userAccount.length() < 4) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"账户名长度小于4");
        }
        //3.密码不小于8
        if (userPassword.length() < 8 || checkPassword.length() < 8) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"密码长度小于8");
        }
        //5. 账户不能包含特殊字符
        String validateStr = "^[a-zA-Z0-9_]+$";
        Matcher matcher = Pattern.compile(validateStr).matcher(userAccount);
        if (!matcher.find()) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"账户包含特殊字符");
        }
        //6. 密码和密码是否相同
        if (!userPassword.equals(checkPassword)) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"密码不相同");
        }
        //4.账户不能重复
        QueryWrapper<User> userQueryWrapper = new QueryWrapper<>();
        userQueryWrapper.eq("account", userAccount);
        long count = this.count(userQueryWrapper);
        if (count > 0) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"账户重复");
        }
        //7. 对密码加密

        String md5Password = DigestUtils.md5DigestAsHex((SALT + userPassword).getBytes());
        //插入数据
        User user = new User();
        user.setAccount(userAccount);
        user.setPassword(md5Password);
        user.setUsername(userAccount + RandomStringUtils.randomAlphanumeric(5));
        boolean saveResult = this.saveOrUpdate(user);
        if (!saveResult) {
            throw new BusinessException(ErrorCode.FLURL,"插入数据失败");
        }
        return user.getId();
    }

    @Override
    public User userLogin(String userAccount, String userPassword, HttpServletRequest request) {
        //1. 校验
        if (StringUtils.isAnyBlank(userAccount, userPassword)) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"参数为空");
        }
        //2.校验账户
        if (userAccount.length() < 4) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"账户名长度小于4");
        }
        //3.密码不小于8
        if (userPassword.length() < 8) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"密码长度小于8");
        }
        //5. 账户不能包含特殊字符
        String validateStr = "^[a-zA-Z0-9_]+$";
        Matcher matcher = Pattern.compile(validateStr).matcher(userAccount);
        if (!matcher.find()) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR,"账户包含特殊字符");
        }
        //7. 对密码加密
        String md5Password = DigestUtils.md5DigestAsHex((SALT + userPassword).getBytes());
        //4.用户是否存在
        QueryWrapper<User> userQueryWrapper = new QueryWrapper<>();
        userQueryWrapper.eq("account", userAccount);
        userQueryWrapper.eq("password", md5Password);
        User user = this.getOne(userQueryWrapper);
        //用户不存在
        if (user == null) {
            log.info("user login failed, userAccount cannot match userPassword");
            throw new BusinessException(ErrorCode.PARAMS_NULL,"用户不存在");
        }
        //8. 用户脱敏
        User safetyUser = getSafeUser(user);
        //9. 记录用户的登录态
        request.getSession().setAttribute(USER_LOGIN_STATE, safetyUser);
        return safetyUser;
    }

    @Override
    public User getSafeUser(User originUser) {
        if (originUser == null) {
            throw new BusinessException(ErrorCode.PARAMS_NULL,"参数为空");
        }
        User safetyUser = new User();
        safetyUser.setId(originUser.getId());
        safetyUser.setUsername(originUser.getUsername());
        safetyUser.setAccount(originUser.getAccount());
        safetyUser.setAvatar(originUser.getAvatar());
        safetyUser.setGender(originUser.getGender());
        safetyUser.setPhone(originUser.getPhone());
        safetyUser.setEmail(originUser.getEmail());
        safetyUser.setStatus(originUser.getStatus());
        safetyUser.setCreateTime(originUser.getCreateTime());
        safetyUser.setAuthority(originUser.getAuthority());
        return safetyUser;
    }

    @Override
    public int userOutLogin(HttpServletRequest request) {
        request.getSession().removeAttribute(USER_LOGIN_STATE);
        return 1;
    }

}




