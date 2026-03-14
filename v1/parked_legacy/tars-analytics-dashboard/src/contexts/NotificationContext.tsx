import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { NotificationData } from '@/types';

interface NotificationContextType {
  notifications: NotificationData[];
  addNotification: (notification: Omit<NotificationData, 'id' | 'timestamp' | 'read'>) => void;
  removeNotification: (id: string) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearAll: () => void;
  unreadCount: number;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

type NotificationAction =
  | { type: 'ADD_NOTIFICATION'; payload: NotificationData }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'MARK_AS_READ'; payload: string }
  | { type: 'MARK_ALL_AS_READ' }
  | { type: 'CLEAR_ALL' };

function notificationReducer(
  state: NotificationData[],
  action: NotificationAction
): NotificationData[] {
  switch (action.type) {
    case 'ADD_NOTIFICATION':
      return [action.payload, ...state];
    case 'REMOVE_NOTIFICATION':
      return state.filter(notification => notification.id !== action.payload);
    case 'MARK_AS_READ':
      return state.map(notification =>
        notification.id === action.payload
          ? { ...notification, read: true }
          : notification
      );
    case 'MARK_ALL_AS_READ':
      return state.map(notification => ({ ...notification, read: true }));
    case 'CLEAR_ALL':
      return [];
    default:
      return state;
  }
}

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [notifications, dispatch] = useReducer(notificationReducer, []);

  const addNotification = useCallback(
    (notification: Omit<NotificationData, 'id' | 'timestamp' | 'read'>) => {
      const newNotification: NotificationData = {
        ...notification,
        id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        read: false,
      };

      dispatch({ type: 'ADD_NOTIFICATION', payload: newNotification });

      // Auto-remove notification after 5 seconds for non-error types
      if (notification.type !== 'error') {
        setTimeout(() => {
          dispatch({ type: 'REMOVE_NOTIFICATION', payload: newNotification.id });
        }, 5000);
      }
    },
    []
  );

  const removeNotification = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', payload: id });
  }, []);

  const markAsRead = useCallback((id: string) => {
    dispatch({ type: 'MARK_AS_READ', payload: id });
  }, []);

  const markAllAsRead = useCallback(() => {
    dispatch({ type: 'MARK_ALL_AS_READ' });
  }, []);

  const clearAll = useCallback(() => {
    dispatch({ type: 'CLEAR_ALL' });
  }, []);

  const unreadCount = notifications.filter(n => !n.read).length;

  const value: NotificationContextType = {
    notifications,
    addNotification,
    removeNotification,
    markAsRead,
    markAllAsRead,
    clearAll,
    unreadCount,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
}
