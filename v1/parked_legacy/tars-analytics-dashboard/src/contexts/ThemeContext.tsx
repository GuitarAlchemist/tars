import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { Theme } from '@/types';

interface ThemeContextType {
  theme: Theme;
  toggleMode: () => void;
  setTheme: (theme: Partial<Theme>) => void;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

type ThemeAction =
  | { type: 'TOGGLE_MODE' }
  | { type: 'SET_THEME'; payload: Partial<Theme> }
  | { type: 'LOAD_THEME'; payload: Theme };

const defaultTheme: Theme = {
  mode: 'light',
  primaryColor: '#3b82f6',
  accentColor: '#64748b',
};

function themeReducer(state: Theme, action: ThemeAction): Theme {
  switch (action.type) {
    case 'TOGGLE_MODE':
      return {
        ...state,
        mode: state.mode === 'light' ? 'dark' : 'light',
      };
    case 'SET_THEME':
      return {
        ...state,
        ...action.payload,
      };
    case 'LOAD_THEME':
      return action.payload;
    default:
      return state;
  }
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, dispatch] = useReducer(themeReducer, defaultTheme);

  // Load theme from localStorage on mount
  useEffect(() => {
    try {
      const savedTheme = localStorage.getItem('tars-theme');
      if (savedTheme) {
        const parsedTheme = JSON.parse(savedTheme);
        dispatch({ type: 'LOAD_THEME', payload: parsedTheme });
      } else {
        // Check system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
          dispatch({ type: 'TOGGLE_MODE' });
        }
      }
    } catch (error) {
      console.warn('Failed to load theme from localStorage:', error);
    }
  }, []);

  // Save theme to localStorage when it changes
  useEffect(() => {
    try {
      localStorage.setItem('tars-theme', JSON.stringify(theme));
    } catch (error) {
      console.warn('Failed to save theme to localStorage:', error);
    }
  }, [theme]);

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;
    
    if (theme.mode === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }

    // Set CSS custom properties for colors
    root.style.setProperty('--primary-color', theme.primaryColor);
    root.style.setProperty('--accent-color', theme.accentColor);
  }, [theme]);

  const toggleMode = () => {
    dispatch({ type: 'TOGGLE_MODE' });
  };

  const setTheme = (newTheme: Partial<Theme>) => {
    dispatch({ type: 'SET_THEME', payload: newTheme });
  };

  const value: ThemeContextType = {
    theme,
    toggleMode,
    setTheme,
    isDark: theme.mode === 'dark',
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
