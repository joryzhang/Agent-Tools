package com.jory.usercenter.common;

/**
 * @Author: Jory Zhang
 * @Date: 2024/5/9 20 41
 * @Description:
 */

/**
 * 全局错误码
 */
public enum ErrorCode {
    SUCCESS(0,"ok",""),
    PARAMS_ERROR(40000,"请求参数错误 ",""),
    PARAMS_NULL(40001,"请求数据为空" ,""),
    NOT_LOGIN(40002,"未登录" ,""),
    NOT_FOUND_ERROR(40400,"请求数据不存在",""),
    FLURL(30000,"失败" ,""),
    SYSTEM_ERROR(50000,"系统内部异常" ,""),
    NO_AUTH(40100,"无权限" ,""),
    NOT_IMPLEMENTED_ERROR(50100,"功能未实现","");

    /**
     * 状态码信息
     */
    private final int code;
    private final String message;
    private final String description;


    ErrorCode(int code, String message, String description) {
        this.code = code;
        this.message = message;
        this.description = description;
    }

    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }

    public String getDescription() {
        return description;
    }
}
