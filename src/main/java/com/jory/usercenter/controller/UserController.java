package com.jory.usercenter.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.jory.usercenter.common.BaseResponse;
import com.jory.usercenter.common.ErrorCode;
import com.jory.usercenter.common.ResultUtils;
import com.jory.usercenter.exception.BusinessException;
import com.jory.usercenter.model.User;
import com.jory.usercenter.model.request.UserLoginRequest;
import com.jory.usercenter.model.request.UserRegisterRequest;
import com.jory.usercenter.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;
import java.util.List;
import java.util.stream.Collectors;

import static com.jory.usercenter.constant.UserConstant.ADMIN_ROLE;
import static com.jory.usercenter.constant.UserConstant.USER_LOGIN_STATE;

/**
 * @Author: Jory Zhang
 * @Date: 2024/4/27 22 00
 * @Description:
 */

@RestController
@RequestMapping(value = "/user",method = {RequestMethod.GET,RequestMethod.POST})
public class UserController {

    @Resource
    private UserService userService;

    @GetMapping("/current")
    public BaseResponse<User> getCurrentUser(HttpServletRequest request) {
        Object userObj = request.getSession().getAttribute(USER_LOGIN_STATE);
        User currentUser = (User) userObj;
        if (currentUser == null) {
            /**
             * 用户未登录
             */
//            return ResultUtils.error(ErrorCode.NOT_LOGIN);
            throw new BusinessException(ErrorCode.NOT_LOGIN, "用户未登录");
        }
        long id = currentUser.getId();
        //todo 校验用户是否合法
        User user = userService.getById(id);
        User result = userService.getSafeUser(user);
        return ResultUtils.success(result);
    }

    @PostMapping("/outLogin")
    public BaseResponse<Integer> outLogin(HttpServletRequest request) {
        if (request == null) {
            throw new BusinessException(ErrorCode.PARAMS_NULL);
        }
        int result = userService.userOutLogin(request);
        return ResultUtils.success(result);
    }

    @PostMapping("/register")
    public BaseResponse<Long> userRegister(@RequestBody UserRegisterRequest userRegisterRequest) {
        if (userRegisterRequest == null) {
            //请求参数错误
            throw new BusinessException(ErrorCode.PARAMS_ERROR, "请求参数错误");
        }
        String userAccount = userRegisterRequest.getUserAccount();
        String userPassword = userRegisterRequest.getUserPassword();
        String checkPassword = userRegisterRequest.getCheckPassword();
        if (StringUtils.isAnyBlank(userAccount, userPassword, checkPassword)) {
            throw new BusinessException(ErrorCode.PARAMS_NULL, "数据为空");
        }
        long result = userService.userRegister(userAccount, userPassword, checkPassword);
        return ResultUtils.success(result);
    }

    @PostMapping("/login")
    public BaseResponse<User> userLogin(@RequestBody UserLoginRequest userLoginRequest, HttpServletRequest httpServletRequest) {
        if (userLoginRequest == null) {
            throw new BusinessException(ErrorCode.PARAMS_NULL);
        }
        String userAccount = userLoginRequest.getUserAccount();
        String userPassword = userLoginRequest.getUserPassword();
        if (StringUtils.isAnyBlank(userAccount, userPassword)) {
            throw new BusinessException(ErrorCode.PARAMS_NULL);
        }
        User result = userService.userLogin(userAccount, userPassword, httpServletRequest);
        return ResultUtils.success(result);
    }

    /**
     * 用户查询
     *
     * @param username
     * @return
     */
    @GetMapping("/search")
    public BaseResponse<List<User>> searchUsers(String username, HttpServletRequest httpServletRequest) {
        //鉴权 仅管理员可以
        if (!roleValid(httpServletRequest)) {
            throw new BusinessException(ErrorCode.NO_AUTH);
        }
        ;

        QueryWrapper<User> userQueryWrapper = new QueryWrapper<>();
        if (StringUtils.isNotBlank(username)) {
            userQueryWrapper.like("username", username);
        }
        List<User> userList = userService.list(userQueryWrapper);
        List<User> result = userList.stream().map(user -> userService.getSafeUser(user)).collect(Collectors.toList());
        return ResultUtils.success(result);
    }

    @PostMapping("/delete")
    public BaseResponse<Boolean> searchUsers(int id, HttpServletRequest httpServletRequest) {
        if (!roleValid(httpServletRequest)) {
            throw new BusinessException(ErrorCode.NO_AUTH);
        }
        ;
        if (id < 0) {
            throw new BusinessException(ErrorCode.PARAMS_NULL, "无此用户信息");
        }
        boolean result = userService.removeById(id);
        return ResultUtils.success(result);
    }

    /**
     * 是否为管理员
     *
     * @param httpServletRequest
     * @return
     */
    private boolean roleValid(HttpServletRequest httpServletRequest) {
        Object userObj = httpServletRequest.getSession().getAttribute(USER_LOGIN_STATE);
        User user = (User) userObj;
        return user != null && user.getAuthority() == ADMIN_ROLE;
    }
}
