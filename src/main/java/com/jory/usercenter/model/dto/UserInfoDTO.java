package com.jory.usercenter.model.dto;

import lombok.Data;
import java.io.Serializable;
import java.util.Date;

/**
 * 用户信息 DTO (供 Python Agent 调用)
 * 不包含敏感信息(密码等)
 */
@Data
public class UserInfoDTO implements Serializable {
    
    /**
     * 用户 ID
     */
    private Long id;
    
    /**
     * 用户名
     */
    private String username;
    
    /**
     * 账号
     */
    private String account;
    
    /**
     * 头像
     */
    private String avatar;
    
    /**
     * 性别 (0-女 1-男)
     */
    private Integer gender;
    
    /**
     * 电话
     */
    private String phone;
    
    /**
     * 邮箱
     */
    private String email;
    
    /**
     * 用户状态 (0-正常 1-禁用)
     */
    private Integer status;
    
    /**
     * 用户角色 (0-普通用户 1-管理员)
     */
    private Integer authority;
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * VIP 等级 (扩展字段,可选)
     */
    private String vipLevel;
    
    /**
     * 账户余额 (扩展字段,可选)
     */
    private Double balance;
    
    private static final long serialVersionUID = 1L;
}
