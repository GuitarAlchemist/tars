// Local Storage Utilities for TARS Analytics Dashboard

const STORAGE_KEYS = {
  AUTH_TOKEN: 'tars_auth_token',
  USER_PREFERENCES: 'tars_user_preferences',
  THEME: 'tars_theme',
  DASHBOARD_LAYOUT: 'tars_dashboard_layout',
  RECENT_SEARCHES: 'tars_recent_searches',
} as const;

// Generic storage utility
class StorageUtil {
  private isAvailable(): boolean {
    try {
      const test = '__storage_test__';
      localStorage.setItem(test, test);
      localStorage.removeItem(test);
      return true;
    } catch {
      return false;
    }
  }

  setItem<T>(key: string, value: T): void {
    if (!this.isAvailable()) {
      console.warn('localStorage is not available');
      return;
    }

    try {
      const serializedValue = JSON.stringify(value);
      localStorage.setItem(key, serializedValue);
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }

  getItem<T>(key: string): T | null {
    if (!this.isAvailable()) {
      console.warn('localStorage is not available');
      return null;
    }

    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return null;
    }
  }

  removeItem(key: string): void {
    if (!this.isAvailable()) {
      console.warn('localStorage is not available');
      return;
    }

    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from localStorage:', error);
    }
  }

  clear(): void {
    if (!this.isAvailable()) {
      console.warn('localStorage is not available');
      return;
    }

    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  }
}

const storage = new StorageUtil();

// Token storage utilities
export const tokenStorage = {
  setToken(token: string): void {
    storage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
  },

  getToken(): string | null {
    return storage.getItem<string>(STORAGE_KEYS.AUTH_TOKEN);
  },

  removeToken(): void {
    storage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
  },

  hasToken(): boolean {
    return !!this.getToken();
  },
};

// User preferences storage
export const preferencesStorage = {
  setPreferences(preferences: Record<string, any>): void {
    storage.setItem(STORAGE_KEYS.USER_PREFERENCES, preferences);
  },

  getPreferences(): Record<string, any> | null {
    return storage.getItem<Record<string, any>>(STORAGE_KEYS.USER_PREFERENCES);
  },

  updatePreference(key: string, value: any): void {
    const current = this.getPreferences() || {};
    current[key] = value;
    this.setPreferences(current);
  },

  removePreferences(): void {
    storage.removeItem(STORAGE_KEYS.USER_PREFERENCES);
  },
};

// Dashboard layout storage
export const layoutStorage = {
  setLayout(layout: any): void {
    storage.setItem(STORAGE_KEYS.DASHBOARD_LAYOUT, layout);
  },

  getLayout(): any | null {
    return storage.getItem(STORAGE_KEYS.DASHBOARD_LAYOUT);
  },

  removeLayout(): void {
    storage.removeItem(STORAGE_KEYS.DASHBOARD_LAYOUT);
  },
};

// Recent searches storage
export const searchStorage = {
  addSearch(query: string): void {
    const searches = this.getSearches();
    const updated = [query, ...searches.filter(s => s !== query)].slice(0, 10);
    storage.setItem(STORAGE_KEYS.RECENT_SEARCHES, updated);
  },

  getSearches(): string[] {
    return storage.getItem<string[]>(STORAGE_KEYS.RECENT_SEARCHES) || [];
  },

  clearSearches(): void {
    storage.removeItem(STORAGE_KEYS.RECENT_SEARCHES);
  },
};

// Session storage utilities (for temporary data)
class SessionStorageUtil {
  private isAvailable(): boolean {
    try {
      const test = '__session_test__';
      sessionStorage.setItem(test, test);
      sessionStorage.removeItem(test);
      return true;
    } catch {
      return false;
    }
  }

  setItem<T>(key: string, value: T): void {
    if (!this.isAvailable()) {
      console.warn('sessionStorage is not available');
      return;
    }

    try {
      const serializedValue = JSON.stringify(value);
      sessionStorage.setItem(key, serializedValue);
    } catch (error) {
      console.error('Error saving to sessionStorage:', error);
    }
  }

  getItem<T>(key: string): T | null {
    if (!this.isAvailable()) {
      console.warn('sessionStorage is not available');
      return null;
    }

    try {
      const item = sessionStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error('Error reading from sessionStorage:', error);
      return null;
    }
  }

  removeItem(key: string): void {
    if (!this.isAvailable()) {
      console.warn('sessionStorage is not available');
      return;
    }

    try {
      sessionStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from sessionStorage:', error);
    }
  }

  clear(): void {
    if (!this.isAvailable()) {
      console.warn('sessionStorage is not available');
      return;
    }

    try {
      window.sessionStorage.clear();
    } catch (error) {
      console.error('Error clearing sessionStorage:', error);
    }
  }
}

export const sessionStorage = new SessionStorageUtil();

// Cache utilities with expiration
export const cacheStorage = {
  setWithExpiry<T>(key: string, value: T, ttlMinutes: number): void {
    const now = new Date();
    const item = {
      value,
      expiry: now.getTime() + ttlMinutes * 60 * 1000,
    };
    storage.setItem(key, item);
  },

  getWithExpiry<T>(key: string): T | null {
    const item = storage.getItem<{ value: T; expiry: number }>(key);
    
    if (!item) {
      return null;
    }

    const now = new Date();
    if (now.getTime() > item.expiry) {
      storage.removeItem(key);
      return null;
    }

    return item.value;
  },
};

export default storage;
