package com.jory.usercenter.controller;

import com.jory.usercenter.common.BaseResponse;
import com.jory.usercenter.common.ErrorCode;
import com.jory.usercenter.common.ResultUtils;
import com.jory.usercenter.exception.BusinessException;
import com.jory.usercenter.model.User;
import com.jory.usercenter.model.dto.UserInfoDTO;
import com.jory.usercenter.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;

/**
 * 内部 API 控制器
 * 供 Python Agent 调用,不对外暴露
 * 
 * 安全策略:
 * 1. 只允许内网访问 (通过 Nginx/网关配置)
 * 2. 或者通过共享密钥验证
 */
@RestController
@RequestMapping("/internal")
@Slf4j
public class InternalApiController {

    @Resource
    private UserService userService;

    /**
     * 获取用户信息 (供 Agent 调用)
     * 
     * @param userId 用户 ID
     * @param request HTTP 请求
     * @return 用户信息 DTO
     */
    @GetMapping("/user/info")
    public BaseResponse<UserInfoDTO> getUserInfo(
            @RequestParam("userId") Long userId,
            HttpServletRequest request
    ) {
        log.info("[Internal API] 获取用户信息: userId={}", userId);
        
        // 参数校验
        if (userId == null || userId <= 0) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR, "用户 ID 无效");
        }
        
        // TODO: 安全校验 - 验证请求来源
        // 方案1: 检查请求来源 IP 是否为内网
        // 方案2: 检查请求头中的共享密钥
        // String apiKey = request.getHeader("X-Internal-API-Key");
        // if (!"your-secret-key".equals(apiKey)) {
        //     throw new BusinessException(ErrorCode.NO_AUTH, "无权访问内部接口");
        // }
        
        // 查询用户
        User user = userService.getById(userId);
        if (user == null) {
            throw new BusinessException(ErrorCode.NOT_FOUND_ERROR, "用户不存在");
        }
        
        // 转换为 DTO (脱敏)
        UserInfoDTO userInfoDTO = new UserInfoDTO();
        BeanUtils.copyProperties(user, userInfoDTO);
        
        // 设置扩展字段 (示例)
        userInfoDTO.setVipLevel(user.getAuthority() == 1 ? "管理员" : "普通用户");
        userInfoDTO.setBalance(0.0); // 示例:从其他服务获取余额
        
        log.info("[Internal API] 返回用户信息: username={}", user.getUsername());
        return ResultUtils.success(userInfoDTO);
    }
    
    /**
     * 批量获取用户信息 (可选)
     * 
     * @param userIds 用户 ID 列表,逗号分隔
     * @param request HTTP 请求
     * @return 用户信息列表
     */
    @GetMapping("/user/batch")
    public BaseResponse<java.util.List<UserInfoDTO>> getBatchUserInfo(
            @RequestParam("userIds") String userIds,
            HttpServletRequest request
    ) {
        log.info("[Internal API] 批量获取用户信息: userIds={}", userIds);
        
        // TODO: 实现批量查询逻辑
        throw new BusinessException(ErrorCode.NOT_IMPLEMENTED_ERROR, "功能开发中");
    }
    
    /**
     * 获取用户统计信息 (示例)
     * 
     * @param userId 用户 ID
     * @param request HTTP 请求
     * @return 统计信息
     */
    @GetMapping("/user/stats")
    public BaseResponse<java.util.Map<String, Object>> getUserStats(
            @RequestParam("userId") Long userId,
            HttpServletRequest request
    ) {
        log.info("[Internal API] 获取用户统计: userId={}", userId);
        
        // 参数校验
        if (userId == null || userId <= 0) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR, "用户 ID 无效");
        }
        
        // 构建统计数据 (示例)
        java.util.Map<String, Object> stats = new java.util.HashMap<>();
        stats.put("userId", userId);
        stats.put("totalOrders", 0); // 示例:从订单服务获取
        stats.put("totalSpent", 0.0); // 示例:从账单服务获取
        stats.put("lastLoginTime", new java.util.Date()); // 示例
        
        return ResultUtils.success(stats);
    }
}
