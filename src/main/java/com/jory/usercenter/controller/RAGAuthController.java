package com.jory.usercenter.controller;

import com.jory.usercenter.common.BaseResponse;
import com.jory.usercenter.common.ErrorCode;
import com.jory.usercenter.common.ResultUtils;
import com.jory.usercenter.exception.BusinessException;
import com.jory.usercenter.model.User;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import static com.jory.usercenter.constant.UserConstant.USER_LOGIN_STATE;

/**
 * RAG AI 服务鉴权接口
 * 用于颁发 Python 端通行的 Token (Hybrid Auth Pattern)
 */
@RestController
@RequestMapping("/auth")
public class RAGAuthController {

    @org.springframework.beans.factory.annotation.Value("${rag.jwt.secret}")
    private String jwtSecret;

    /**
     * 获取 AI 服务的 Access Token
     * 前端调用此接口，获取 token，然后放入 Header: Authorization: Bearer {token} 去访问 Python 服务
     */
    @GetMapping("/ai-token")
    public BaseResponse<String> getAIToken(HttpServletRequest request) {
        // 1. 检查 Session 登录状态
        HttpSession session = request.getSession();
        Object userObj = session.getAttribute(USER_LOGIN_STATE);
        User currentUser = (User) userObj;

        if (currentUser == null) {
            throw new BusinessException(ErrorCode.NOT_LOGIN, "请先登录使用的AI助手");
        }

        // 2. 签发 JWT Token
        // 有效期 10 分钟 (短效令牌，安全性高)
        long expirationTime = 10 * 60 * 1000; 
        
        Map<String, Object> claims = new HashMap<>();
        // 注意：claims 会覆盖 setSubject 等设置，所以使用 setClaims 后如果还要标准字段，需要小心
        
        // 推荐做法：
        String token = Jwts.builder()
                .setSubject(String.valueOf(currentUser.getId())) // sub: user_id
                .claim("role", currentUser.getAuthority())       // custom claim
                .claim("username", currentUser.getUsername()) // custom claim
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + expirationTime))
                .signWith(SignatureAlgorithm.HS256, jwtSecret.getBytes()) // 使用注入的密钥
                .compact();

        return ResultUtils.success(token);
    }
}
