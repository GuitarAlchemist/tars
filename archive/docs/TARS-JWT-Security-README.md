# 🔐 TARS JWT Security Implementation

**Comprehensive JWT Authentication and Authorization for TARS Endpoints**

## 🎯 Overview

TARS now includes robust JWT (JSON Web Token) authentication to protect all endpoints from malicious activity. This implementation provides:

- **JWT Token Generation**: Secure token creation with configurable expiration
- **Token Validation**: Comprehensive token verification and user context extraction
- **Role-Based Access**: Support for different user roles (User, Agent, Administrator, System)
- **Middleware Protection**: Automatic endpoint protection with configurable anonymous access
- **Security Headers**: Standard security headers for enhanced protection
- **Audit Logging**: Comprehensive security event logging

## 🚀 Quick Start

### 1. Authentication Endpoints

#### Login
```bash
POST /api/auth/login
Content-Type: application/json

{
  "Username": "admin",
  "Password": "admin123",
  "Role": "Administrator"  // Optional
}
```

**Response:**
```json
{
  "Token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "ExpiresAt": "2024-01-01T12:00:00Z",
  "User": {
    "UserId": "admin",
    "Username": "Administrator",
    "Role": "Administrator",
    "Permissions": ["read", "write"]
  }
}
```

#### Token Validation
```bash
POST /api/auth/validate
Content-Type: application/json

{
  "Token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 2. Using JWT Tokens

#### Authorization Header (Recommended)
```bash
GET /api/service/status
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Query Parameter (Fallback)
```bash
GET /api/service/status?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🔧 Configuration

### Security Configuration File
Location: `Configuration/security.config.yaml`

```yaml
TarsSecurity:
  # Authentication settings
  EnableAuthentication: true
  EnableAuthorization: true
  DefaultAuthType: "JWT"
  RequireHttps: false  # Set to true in production
  AllowAnonymous: false
  
  # JWT Configuration
  JwtSecret: "CHANGE_THIS_IN_PRODUCTION_TO_A_SECURE_SECRET_KEY"
  JwtIssuer: "TARS"
  JwtAudience: "TARS-API"
  JwtExpirationMinutes: 60
  
  # CORS Configuration
  EnableCors: true
  AllowedOrigins:
    - "http://localhost:3000"
    - "http://localhost:8080"
  AllowedMethods:
    - "GET"
    - "POST"
    - "PUT"
    - "DELETE"
  AllowedHeaders:
    - "Content-Type"
    - "Authorization"
```

### Environment Variables (Production)
```bash
# Set secure JWT secret
JWT_SECRET=your-super-secure-secret-key-here

# Enable HTTPS
REQUIRE_HTTPS=true

# Restrict CORS origins
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## 👥 User Roles and Permissions

### Default Test Users
| Username | Password | Role | Description |
|----------|----------|------|-------------|
| `admin` | `admin123` | Administrator | Full system access |
| `user` | `user123` | User | Basic read/write access |
| `agent` | `agent123` | Agent | Agent operations access |
| `system` | `system123` | System | System-level access |

### Role Hierarchy
1. **System** - Full administrative access, all permissions
2. **Administrator** - Management access, most permissions
3. **Agent** - Agent operations, limited permissions
4. **User** - Basic access, read/write permissions
5. **Anonymous** - No authentication, very limited access

## 🛡️ Security Features

### JWT Token Security
- **HS256 Algorithm**: Industry-standard HMAC SHA-256 signing
- **Configurable Expiration**: Default 1 hour, customizable
- **Secure Claims**: User ID, username, role, and permissions
- **Clock Skew Tolerance**: 5-minute tolerance for time differences

### Endpoint Protection
- **Automatic Middleware**: All endpoints protected by default
- **Anonymous Endpoints**: Health checks, documentation, login
- **Role-Based Access**: Endpoints can require specific roles
- **Permission Checks**: Fine-grained permission validation

### Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Strict-Transport-Security: max-age=31536000; includeSubDomains (HTTPS only)
```

## 🔍 Testing JWT Authentication

### PowerShell Test Script
Run the included test script to verify JWT functionality:

```powershell
.\test-jwt-auth.ps1
```

This script tests:
- ✅ Valid user login for all roles
- ✅ Token validation
- ✅ Invalid login rejection
- ✅ Invalid token rejection

### Manual Testing with curl

#### Login
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"Username":"admin","Password":"admin123"}'
```

#### Access Protected Endpoint
```bash
curl -X GET http://localhost:5000/api/service/status \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🚨 Security Best Practices

### Production Deployment
1. **Change Default Secret**: Use a strong, unique JWT secret
2. **Enable HTTPS**: Set `RequireHttps: true`
3. **Restrict CORS**: Specify exact allowed origins
4. **Strong Passwords**: Implement proper user authentication
5. **Token Rotation**: Consider implementing refresh tokens
6. **Audit Logging**: Monitor authentication events

### JWT Secret Generation
```powershell
# Generate secure 512-bit secret
$bytes = New-Object byte[] 64
[System.Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($bytes)
[Convert]::ToBase64String($bytes)
```

### Environment-Specific Configuration

#### Development
```yaml
TarsSecurity:
  RequireHttps: false
  AllowAnonymous: true
  JwtExpirationMinutes: 480  # 8 hours for development
  AllowedOrigins: ["*"]
```

#### Production
```yaml
TarsSecurity:
  RequireHttps: true
  AllowAnonymous: false
  JwtExpirationMinutes: 30   # 30 minutes for security
  AllowedOrigins: ["https://yourdomain.com"]
```

## 🔧 Troubleshooting

### Common Issues

#### 401 Unauthorized
- **Cause**: Missing or invalid JWT token
- **Solution**: Ensure valid token in Authorization header

#### 403 Forbidden
- **Cause**: Valid token but insufficient permissions
- **Solution**: Check user role and endpoint requirements

#### Token Expired
- **Cause**: JWT token past expiration time
- **Solution**: Re-authenticate to get new token

#### CORS Errors
- **Cause**: Origin not in allowed list
- **Solution**: Add origin to `AllowedOrigins` configuration

### Debug Logging
Enable debug logging to troubleshoot authentication issues:

```yaml
Logging:
  LogLevel:
    TarsEngine.FSharp.WindowsService.Security: Debug
```

## 📊 Security Monitoring

### Authentication Events
The system logs the following security events:
- ✅ Successful authentication
- ❌ Failed authentication attempts
- ⏰ Token expiration
- 🚫 Invalid token attempts
- 🔒 Authorization failures

### Log Locations
- **Console**: Real-time logging output
- **File**: `logs/security-audit.log` (if configured)
- **Windows Event Log**: Application log entries

## 🚀 Next Steps

### Enhanced Security Features
1. **Refresh Tokens**: Implement token refresh mechanism
2. **API Keys**: Add API key authentication for services
3. **Rate Limiting**: Implement request rate limiting
4. **Multi-Factor Auth**: Add MFA support
5. **OAuth2 Integration**: Support external identity providers

### Integration Examples
1. **Agent Authentication**: Agents can authenticate with JWT tokens
2. **Service-to-Service**: Internal services use JWT for communication
3. **Web UI**: Frontend applications authenticate users
4. **API Clients**: External applications access TARS APIs securely

## 🎯 Summary

TARS now provides enterprise-grade JWT authentication that:

- ✅ **Protects all endpoints** from unauthorized access
- ✅ **Supports multiple user roles** with appropriate permissions
- ✅ **Provides secure token generation** and validation
- ✅ **Includes comprehensive logging** for security monitoring
- ✅ **Follows security best practices** with configurable options
- ✅ **Enables easy testing** with included test scripts

The JWT implementation ensures that TARS can be safely deployed in production environments while maintaining the flexibility needed for development and testing.

---

**🤖 TARS - Autonomous Development Platform**  
*Secure, scalable, and intelligent software development automation*
