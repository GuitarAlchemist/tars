import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  Badge
} from '@mui/material';
import {
  Whatshot as FireIcon,
  Update as UpdateIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon
} from '@mui/icons-material';
// import { format } from 'date-fns';

// Mock date formatting function for demo
const format = (date: Date, formatStr: string): string => {
  if (formatStr === 'MMM dd, yyyy HH:mm:ss') {
    return date.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' }) + ' ' +
           date.toLocaleTimeString('en-US', { hour12: false });
  }
  if (formatStr === 'HH:mm:ss') {
    return date.toLocaleTimeString('en-US', { hour12: false });
  }
  return date.toLocaleString();
};

interface HeaderProps {
  fireCount: number;
  lastUpdate: Date;
  isLoading: boolean;
}

const Header: React.FC<HeaderProps> = ({ fireCount, lastUpdate, isLoading }) => {
  const formatLastUpdate = (date: Date): string => {
    return format(date, 'MMM dd, yyyy HH:mm:ss');
  };

  const getFireCountColor = (count: number): 'default' | 'warning' | 'error' => {
    if (count > 100) return 'error';
    if (count > 50) return 'warning';
    return 'default';
  };

  return (
    <AppBar 
      position="static" 
      sx={{ 
        background: 'linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%)',
        boxShadow: '0 2px 10px rgba(255, 107, 53, 0.3)',
        borderBottom: '1px solid rgba(255, 107, 53, 0.2)'
      }}
    >
      <Toolbar sx={{ minHeight: '64px !important', px: 3 }}>
        {/* Logo and Title */}
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <FireIcon 
            sx={{ 
              fontSize: 32, 
              color: '#ff6b35', 
              mr: 2,
              animation: isLoading ? 'pulse 2s infinite' : 'none',
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.7 }
              }
            }} 
          />
          <Box>
            <Typography 
              variant="h5" 
              component="h1" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(45deg, #ff6b35, #f7931e)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '0.5px'
              }}
            >
              Forest Fire Monitor
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                color: '#b0b0b0',
                display: 'block',
                lineHeight: 1,
                fontSize: '11px'
              }}
            >
              Real-time Fire Tracking • Powered by TARS AI
            </Typography>
          </Box>
        </Box>

        {/* Status Information */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Active Fires Count */}
          <Tooltip title="Total active fires detected">
            <Chip
              icon={<FireIcon />}
              label={`${fireCount} Active Fires`}
              color={getFireCountColor(fireCount)}
              variant="filled"
              sx={{
                fontWeight: 600,
                '& .MuiChip-icon': {
                  color: 'inherit'
                }
              }}
            />
          </Tooltip>

          {/* Last Update */}
          <Tooltip title={`Last updated: ${formatLastUpdate(lastUpdate)}`}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <UpdateIcon 
                sx={{ 
                  fontSize: 16, 
                  color: isLoading ? '#ff6b35' : '#4caf50',
                  animation: isLoading ? 'spin 2s linear infinite' : 'none',
                  '@keyframes spin': {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' }
                  }
                }} 
              />
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isLoading ? '#ff6b35' : '#4caf50',
                  fontWeight: 500,
                  fontSize: '12px'
                }}
              >
                {isLoading ? 'Updating...' : format(lastUpdate, 'HH:mm:ss')}
              </Typography>
            </Box>
          </Tooltip>

          {/* Data Sources */}
          <Tooltip title="Data sources: NASA FIRMS, Canadian Wildfire Information System">
            <Chip
              label="Multi-Source"
              size="small"
              variant="outlined"
              sx={{
                borderColor: '#f7931e',
                color: '#f7931e',
                fontSize: '11px'
              }}
            />
          </Tooltip>

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
            <Tooltip title="Notifications">
              <IconButton size="small" sx={{ color: '#b0b0b0' }}>
                <Badge badgeContent={2} color="error">
                  <NotificationsIcon fontSize="small" />
                </Badge>
              </IconButton>
            </Tooltip>

            <Tooltip title="Information">
              <IconButton size="small" sx={{ color: '#b0b0b0' }}>
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            <Tooltip title="Settings">
              <IconButton size="small" sx={{ color: '#b0b0b0' }}>
                <SettingsIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Toolbar>

      {/* Status Bar */}
      <Box
        sx={{
          background: 'rgba(255, 107, 53, 0.1)',
          borderTop: '1px solid rgba(255, 107, 53, 0.2)',
          px: 3,
          py: 0.5
        }}
      >
        <Typography 
          variant="caption" 
          sx={{ 
            color: '#b0b0b0',
            fontSize: '11px',
            display: 'flex',
            alignItems: 'center',
            gap: 2
          }}
        >
          <span>🌍 Coverage: United States & Canada</span>
          <span>•</span>
          <span>🛰️ Satellite: VIIRS, MODIS</span>
          <span>•</span>
          <span>⏱️ Update Frequency: 15 minutes</span>
          <span>•</span>
          <span>🎯 Resolution: 375m</span>
        </Typography>
      </Box>
    </AppBar>
  );
};

export default Header;
