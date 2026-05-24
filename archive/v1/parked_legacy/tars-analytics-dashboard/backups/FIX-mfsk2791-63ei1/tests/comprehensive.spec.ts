import { test, expect } from '@playwright/test';
import { injectAxe, checkA11y } from 'axe-playwright';

test.describe('TARS Analytics Dashboard - Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load login page and authenticate successfully', async ({ page }) => {
    // Should redirect to login page when not authenticated
    await expect(page).toHaveURL('/login');
    
    // Check login page elements
    await expect(page.locator('h2')).toContainText('Welcome to TARS');
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    
    // Test login with valid credentials
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard after successful login
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('h1')).toContainText('Dashboard');
  });

  test('should display dashboard with metrics and charts', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Wait for dashboard to load
    await expect(page).toHaveURL('/dashboard');
    
    // Check metric cards are present
    await expect(page.locator('text=Total Revenue')).toBeVisible();
    await expect(page.locator('text=Total Users')).toBeVisible();
    await expect(page.locator('text=Active Users')).toBeVisible();
    await expect(page.locator('text=Conversion Rate')).toBeVisible();
    
    // Check charts are loading
    await expect(page.locator('text=Revenue Trend')).toBeVisible();
    await expect(page.locator('text=User Growth')).toBeVisible();
    await expect(page.locator('text=Real-time Analytics')).toBeVisible();
    
    // Wait for data to load (mock API has delays)
    await page.waitForTimeout(2000);
    
    // Check that metric values are displayed
    await expect(page.locator('text=$')).toBeVisible(); // Revenue values
    await expect(page.locator('text=%')).toBeVisible(); // Percentage values
  });

  test('should navigate between pages correctly', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Test navigation to Users page
    await page.click('text=Users');
    await expect(page).toHaveURL('/users');
    await expect(page.locator('h1')).toContainText('Users');
    
    // Test navigation to Reports page
    await page.click('text=Reports');
    await expect(page).toHaveURL('/reports');
    await expect(page.locator('h1')).toContainText('Reports');
    
    // Test navigation to Settings page
    await page.click('text=Settings');
    await expect(page).toHaveURL('/settings');
    await expect(page.locator('h1')).toContainText('Settings');
    
    // Test navigation back to Dashboard
    await page.click('text=Dashboard');
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('h1')).toContainText('Dashboard');
  });

  test('should have responsive design', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Check mobile menu button is visible
    await expect(page.locator('button[aria-label="Open sidebar"]')).toBeVisible();
    
    // Check that content adapts to mobile
    const container = page.locator('.max-w-7xl');
    const boundingBox = await container.boundingBox();
    expect(boundingBox?.width).toBeLessThanOrEqual(375);
    
    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    
    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    
    // Check sidebar is visible on desktop
    await expect(page.locator('text=TARS')).toBeVisible();
  });

  test('should handle user interactions correctly', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Test theme toggle
    const themeButton = page.locator('button[aria-label*="mode"]');
    await themeButton.click();
    
    // Check if dark mode is applied (this might fail intentionally)
    await page.waitForTimeout(500);
    
    // Test refresh button on dashboard
    await page.click('text=Refresh');
    
    // Test search functionality
    await page.fill('input[placeholder="Search..."]', 'test query');
    await expect(page.locator('input[placeholder="Search..."]')).toHaveValue('test query');
    
    // Test user menu
    await page.click('button:has-text("TARS Administrator")');
    await expect(page.locator('text=Sign out')).toBeVisible();
  });

  test('should have good performance', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    
    const startTime = Date.now();
    await page.click('button[type="submit"]');
    
    // Wait for dashboard to fully load
    await expect(page.locator('h1')).toContainText('Dashboard');
    await page.waitForTimeout(1000); // Wait for charts to load
    
    const loadTime = Date.now() - startTime;
    
    // Performance should be reasonable (this might fail intentionally)
    expect(loadTime).toBeLessThan(5000); // 5 seconds max
    
    // Check for console errors
    const logs = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        logs.push(msg.text());
      }
    });
    
    await page.reload();
    await page.waitForTimeout(2000);
    
    // Should have minimal console errors (this might fail intentionally)
    expect(logs.length).toBeLessThan(3);
  });

  test('should be accessible', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Inject axe for accessibility testing
    await injectAxe(page);
    
    // Check accessibility on dashboard
    await checkA11y(page, undefined, {
      detailedReport: true,
      detailedReportOptions: { html: true },
    });
    
    // Check for proper ARIA labels
    const ariaElements = await page.locator('[aria-label]').count();
    expect(ariaElements).toBeGreaterThanOrEqual(3);
    
    // Check for alt text on images (this might fail intentionally)
    const images = page.locator('img');
    const imageCount = await images.count();
    
    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      if (await img.isVisible()) {
        const alt = await img.getAttribute('alt');
        expect(alt).toBeTruthy(); // This will fail - no alt text
      }
    }
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Monitor console errors
    const errors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Trigger potential error scenarios
    await page.goto('/nonexistent-page');
    await expect(page.locator('text=404')).toBeVisible();
    
    // Go back to dashboard
    await page.goto('/dashboard');
    
    // Should handle error gracefully (this will detect unhandled errors)
    await page.waitForTimeout(500);
    expect(errors.length).toBe(0); // This will fail due to unhandled errors
  });

  test('should handle async operations correctly', async ({ page }) => {
    // Monitor promise rejections
    const rejections = [];
    page.on('pageerror', error => {
      rejections.push(error.message);
    });
    
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Wait for async operations to complete
    await page.waitForTimeout(3000);
    
    // Should handle promise rejections (this will fail)
    expect(rejections.length).toBe(0);
  });

  test('should load within reasonable time', async ({ page }) => {
    const startTime = Date.now();
    
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Wait for loading to complete
    await page.waitForSelector('#loading.hidden', { timeout: 5000 });
    
    const loadTime = Date.now() - startTime;
    
    // Should load quickly (this might fail due to intentional delays)
    expect(loadTime).toBeLessThan(3000);
  });

  test('should work across different browsers', async ({ page, browserName }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Basic functionality should work in all browsers
    await expect(page.locator('h1')).toContainText('Dashboard');
    
    // Browser-specific checks
    if (browserName === 'webkit') {
      // Safari-specific tests
      await expect(page.locator('text=TARS')).toBeVisible();
    } else if (browserName === 'firefox') {
      // Firefox-specific tests
      await expect(page.locator('text=Analytics Dashboard')).toBeVisible();
    }
    
    // Test common functionality across browsers
    await page.click('text=Users');
    await expect(page).toHaveURL('/users');
  });

  test('should maintain functionality on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@tars.ai');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    
    // Test mobile navigation
    await page.click('button[aria-label="Open sidebar"]');
    await page.click('text=Users');
    await expect(page).toHaveURL('/users');
    
    // Test mobile interactions
    await page.click('text=Add User');
    await expect(page.locator('text=Add New User')).toBeVisible();
  });
});
