import React, { useState } from 'react';
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  Divider,
  Card,
  CardContent,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  LinearProgress,
  Tabs,
  Tab,
  Badge
} from '@mui/material';
import {
  Close as CloseIcon,
  Whatshot as FireIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  LocationOn as LocationIcon,
  Schedule as ScheduleIcon,
  Satellite as SatelliteIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { FirePanelProps } from '../../types/FireTypes';
// import { format } from 'date-fns';

// Mock date formatting function for demo
const format = (date: Date | string, formatStr: string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  if (formatStr === 'MMMM dd, yyyy') {
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: '2-digit' });
  }
  if (formatStr === 'MMM dd, HH:mm') {
    return d.toLocaleDateString('en-US', { month: 'short', day: '2-digit' }) + ', ' +
           d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  }
  return d.toLocaleString();
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div hidden={value !== index} style={{ height: '100%' }}>
      {value === index && <Box sx={{ p: 2, height: '100%' }}>{children}</Box>}
    </div>
  );
};

const FirePanel: React.FC<FirePanelProps> = ({
  selectedFire,
  statistics,
  alerts,
  onClose,
  onAlertDismiss
}) => {
  const [tabValue, setTabValue] = useState(0);
  const isOpen = !!selectedFire || tabValue !== 0;

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getIntensityColor = (intensity: string) => {
    switch (intensity) {
      case 'extreme': return '#8B0000';
      case 'high': return '#FF0000';
      case 'medium': return '#FFA500';
      case 'low': return '#FFD700';
      default: return '#FFA500';
    }
  };

  const getIntensityIcon = (intensity: string) => {
    switch (intensity) {
      case 'extreme': return '🔥🔥🔥';
      case 'high': return '🔥🔥';
      case 'medium': return '🔥';
      case 'low': return '🟡';
      default: return '🔥';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <WarningIcon sx={{ color: '#f44336' }} />;
      case 'warning': return <WarningIcon sx={{ color: '#ff9800' }} />;
      case 'info': return <InfoIcon sx={{ color: '#2196f3' }} />;
      default: return <InfoIcon />;
    }
  };

  return (
    <Drawer
      anchor="right"
      open={isOpen}
      variant="persistent"
      sx={{
        width: isOpen ? 400 : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 400,
          background: 'linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%)',
          borderLeft: '1px solid rgba(255, 107, 53, 0.2)',
          boxShadow: '-5px 0 15px rgba(0, 0, 0, 0.3)'
        }
      }}
    >
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ p: 2, borderBottom: '1px solid rgba(255, 107, 53, 0.2)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6" sx={{ color: '#ff6b35', fontWeight: 600 }}>
              Fire Information
            </Typography>
            <IconButton onClick={onClose} size="small" sx={{ color: '#b0b0b0' }}>
              <CloseIcon />
            </IconButton>
          </Box>
          
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            sx={{
              mt: 1,
              '& .MuiTab-root': {
                color: '#b0b0b0',
                minWidth: 'auto',
                fontSize: '12px'
              },
              '& .Mui-selected': {
                color: '#ff6b35 !important'
              },
              '& .MuiTabs-indicator': {
                backgroundColor: '#ff6b35'
              }
            }}
          >
            <Tab label="Details" />
            <Tab 
              label={
                <Badge badgeContent={statistics.total} color="error" max={999}>
                  Statistics
                </Badge>
              } 
            />
            <Tab 
              label={
                <Badge badgeContent={alerts.length} color="warning" max={99}>
                  Alerts
                </Badge>
              } 
            />
          </Tabs>
        </Box>

        {/* Content */}
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          {/* Fire Details Tab */}
          <TabPanel value={tabValue} index={0}>
            {selectedFire ? (
              <Box>
                {/* Fire Header */}
                <Card sx={{ mb: 2, background: 'rgba(255, 107, 53, 0.1)' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h4" sx={{ mr: 1 }}>
                        {getIntensityIcon(selectedFire.intensity)}
                      </Typography>
                      <Box>
                        <Typography variant="h6" sx={{ color: 'white' }}>
                          Fire {selectedFire.id}
                        </Typography>
                        <Chip
                          label={selectedFire.intensity.toUpperCase()}
                          size="small"
                          sx={{
                            backgroundColor: getIntensityColor(selectedFire.intensity),
                            color: 'white',
                            fontWeight: 600
                          }}
                        />
                      </Box>
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LocationIcon sx={{ fontSize: 16, color: '#b0b0b0' }} />
                      <Typography variant="body2" sx={{ color: '#b0b0b0' }}>
                        {selectedFire.location}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>

                {/* Fire Metrics */}
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Card sx={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" sx={{ color: '#b0b0b0' }}>
                          Fire Radiative Power
                        </Typography>
                        <Typography variant="h6" sx={{ color: '#ff6b35' }}>
                          {selectedFire.frp.toFixed(1)} MW
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card sx={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" sx={{ color: '#b0b0b0' }}>
                          Confidence
                        </Typography>
                        <Typography variant="h6" sx={{ color: '#4caf50' }}>
                          {selectedFire.confidence.toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card sx={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" sx={{ color: '#b0b0b0' }}>
                          Brightness
                        </Typography>
                        <Typography variant="h6" sx={{ color: '#f7931e' }}>
                          {selectedFire.brightness.toFixed(1)} K
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card sx={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                      <CardContent sx={{ p: 2 }}>
                        <Typography variant="caption" sx={{ color: '#b0b0b0' }}>
                          Detection Time
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'white' }}>
                          {selectedFire.acq_time}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {/* Additional Details */}
                <Card sx={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ color: '#ff6b35', mb: 2 }}>
                      Detection Details
                    </Typography>
                    
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <SatelliteIcon sx={{ color: '#b0b0b0', fontSize: 20 }} />
                        </ListItemIcon>
                        <ListItemText
                          primary="Satellite"
                          secondary={selectedFire.satellite}
                          primaryTypographyProps={{ variant: 'caption', color: '#b0b0b0' }}
                          secondaryTypographyProps={{ color: 'white' }}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          <ScheduleIcon sx={{ color: '#b0b0b0', fontSize: 20 }} />
                        </ListItemIcon>
                        <ListItemText
                          primary="Detection Date"
                          secondary={format(new Date(selectedFire.acq_date), 'MMMM dd, yyyy')}
                          primaryTypographyProps={{ variant: 'caption', color: '#b0b0b0' }}
                          secondaryTypographyProps={{ color: 'white' }}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          <AssessmentIcon sx={{ color: '#b0b0b0', fontSize: 20 }} />
                        </ListItemIcon>
                        <ListItemText
                          primary="Data Source"
                          secondary={selectedFire.source}
                          primaryTypographyProps={{ variant: 'caption', color: '#b0b0b0' }}
                          secondaryTypographyProps={{ color: 'white' }}
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Box>
            ) : (
              <Box sx={{ textAlign: 'center', mt: 4 }}>
                <FireIcon sx={{ fontSize: 64, color: '#ff6b35', opacity: 0.5 }} />
                <Typography variant="h6" sx={{ color: '#b0b0b0', mt: 2 }}>
                  Select a fire on the map
                </Typography>
                <Typography variant="body2" sx={{ color: '#666', mt: 1 }}>
                  Click on any fire marker to view detailed information
                </Typography>
              </Box>
            )}
          </TabPanel>

          {/* Statistics Tab */}
          <TabPanel value={tabValue} index={1}>
            <Box>
              {/* Overview Stats */}
              <Card sx={{ mb: 2, background: 'rgba(255, 107, 53, 0.1)' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#ff6b35', mb: 2 }}>
                    Fire Overview
                  </Typography>
                  <Typography variant="h3" sx={{ color: 'white', fontWeight: 700 }}>
                    {statistics.total}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#b0b0b0' }}>
                    Total Active Fires
                  </Typography>
                </CardContent>
              </Card>

              {/* Intensity Breakdown */}
              <Card sx={{ mb: 2, background: 'rgba(255, 255, 255, 0.05)' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ color: '#ff6b35', mb: 2 }}>
                    By Intensity
                  </Typography>
                  
                  {Object.entries(statistics.byIntensity).map(([intensity, count]) => (
                    <Box key={intensity} sx={{ mb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="caption" sx={{ color: '#b0b0b0' }}>
                          {intensity.charAt(0).toUpperCase() + intensity.slice(1)}
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'white' }}>
                          {count}
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={(count / statistics.total) * 100}
                        sx={{
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getIntensityColor(intensity)
                          }
                        }}
                      />
                    </Box>
                  ))}
                </CardContent>
              </Card>

              {/* Source Breakdown */}
              <Card sx={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ color: '#ff6b35', mb: 2 }}>
                    By Data Source
                  </Typography>
                  
                  {Object.entries(statistics.bySource).map(([source, count]) => (
                    count > 0 && (
                      <Box key={source} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2" sx={{ color: '#b0b0b0' }}>
                          {source.toUpperCase()}
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'white', fontWeight: 600 }}>
                          {count}
                        </Typography>
                      </Box>
                    )
                  ))}
                </CardContent>
              </Card>
            </Box>
          </TabPanel>

          {/* Alerts Tab */}
          <TabPanel value={tabValue} index={2}>
            <Box>
              {alerts.length > 0 ? (
                alerts.map((alert) => (
                  <Alert
                    key={alert.id}
                    severity={alert.severity}
                    icon={getSeverityIcon(alert.severity)}
                    onClose={() => onAlertDismiss?.(alert.id)}
                    sx={{
                      mb: 2,
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                      '& .MuiAlert-message': {
                        color: 'white'
                      }
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {alert.message}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#b0b0b0' }}>
                      {alert.location} • {format(alert.timestamp, 'MMM dd, HH:mm')}
                    </Typography>
                  </Alert>
                ))
              ) : (
                <Box sx={{ textAlign: 'center', mt: 4 }}>
                  <InfoIcon sx={{ fontSize: 64, color: '#4caf50', opacity: 0.5 }} />
                  <Typography variant="h6" sx={{ color: '#b0b0b0', mt: 2 }}>
                    No Active Alerts
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#666', mt: 1 }}>
                    All systems are operating normally
                  </Typography>
                </Box>
              )}
            </Box>
          </TabPanel>
        </Box>
      </Box>
    </Drawer>
  );
};

export default FirePanel;
