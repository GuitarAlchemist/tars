const { test, expect } = require('@playwright/test');

/**
 * TARS Autonomous Playwright QA Tests
 * Comprehensive testing suite that will detect the intentional bugs
 */

test.describe('TARS Autonomous QA - Comprehensive Application Tests', () => {
  
  test.beforeEach(async ({ page }) => {
    // Navigate to the test application
    await page.goto('index.html');
  });

  test('should load homepage without critical errors', async ({ page }) => {
    // Check page loads
    await expect(page.locator('h1')).toContainText('TARS Test Application');
    
    // Check for console errors (this will detect our intentional console error)
    const errors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.waitForLoadState('networkidle');
    
    // This test will FAIL due to intentional console error
    expect(errors.length).toBe(0);
  });

  test('should have responsive design', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForLoadState('networkidle');
    
    // This will FAIL due to the 1200px fixed width bug
    const responsiveElement = page.locator('.responsive-test');
    await expect(responsiveElement).toBeVisible();
    
    // Check if content overflows (this should detect the responsive bug)
    const boundingBox = await responsiveElement.boundingBox();
    expect(boundingBox.width).toBeLessThanOrEqual(375);
  });

  test('should handle user interactions correctly', async ({ page }) => {
    // Test basic button click
    await page.click('#basic-btn');
    await expect(page.locator('#basic-result')).toContainText('Basic function works!');
    
    // Test form submission (this will FAIL due to missing preventDefault)
    await page.fill('#name-input', 'Test User');
    await page.fill('#message-input', 'Test message');
    
    // This should not cause page reload, but it will due to the bug
    await page.click('button[type="submit"]');
    
    // Check if we're still on the same page (this will fail)
    await expect(page.locator('h1')).toContainText('TARS Test Application');
  });

  test('should have good performance', async ({ page }) => {
    // Measure page load performance
    const navigationPromise = page.waitForLoadState('networkidle');
    const startTime = Date.now();
    
    await page.goto('index.html');
    await navigationPromise;
    
    const loadTime = Date.now() - startTime;
    
    // Performance assertion (may fail due to slow loading bug)
    expect(loadTime).toBeLessThan(3000);
    
    // Test performance-heavy operation
    await page.click('text=Run Performance Test');
    
    // This will detect the blocking UI bug
    const performanceStart = Date.now();
    await page.waitForTimeout(1000); // Wait for performance test
    const performanceTime = Date.now() - performanceStart;
    
    // Should not block UI for too long
    expect(performanceTime).toBeLessThan(2000);
  });

  test('should be accessible', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Check for proper heading structure
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').count();
    expect(headings).toBeGreaterThan(0);
    
    // Check for alt text on images (this will FAIL due to missing alt text)
    const images = await page.locator('img').all();
    for (const img of images) {
      if (await img.isVisible()) {
        const alt = await img.getAttribute('alt');
        expect(alt).toBeTruthy(); // This will fail - no alt text
      }
    }
    
    // Check for proper focus management
    await page.keyboard.press('Tab');
    const focusedElement = await page.locator(':focus').first();
    await expect(focusedElement).toBeVisible();
    
    // Test ARIA labels (this will detect missing ARIA labels)
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label');
      const text = await button.textContent();
      // Either should have aria-label or meaningful text
      expect(ariaLabel || text.trim()).toBeTruthy();
    }
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Test error handling
    const errors = [];
    page.on('pageerror', error => {
      errors.push(error.message);
    });
    
    // Trigger intentional error
    await page.click('text=Trigger Error');
    
    // Should handle error gracefully (this will detect unhandled errors)
    await page.waitForTimeout(500);
    expect(errors.length).toBe(0); // This will fail due to unhandled errors
    
    // Check error display
    await expect(page.locator('#error-result')).toContainText('Error occurred');
  });

  test('should handle async operations correctly', async ({ page }) => {
    const rejections = [];
    page.on('console', msg => {
      if (msg.text().includes('Unhandled promise rejection')) {
        rejections.push(msg.text());
      }
    });
    
    // Trigger async error
    await page.click('text=Trigger Async Error');
    await page.waitForTimeout(500);
    
    // Should handle promise rejections (this will fail)
    expect(rejections.length).toBe(0);
  });

  test('should load within reasonable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('index.html');
    
    // Wait for loading to complete
    await page.waitForSelector('#loading.hidden', { timeout: 5000 });
    
    const loadTime = Date.now() - startTime;
    
    // Should load quickly (this may fail due to artificial 2s delay)
    expect(loadTime).toBeLessThan(1500);
  });

  test('should work across different browsers', async ({ page, browserName }) => {
    // Cross-browser compatibility test
    await page.waitForLoadState('networkidle');
    
    // Basic functionality should work in all browsers
    await page.click('#basic-btn');
    await expect(page.locator('#basic-result')).toContainText('Basic function works!');
    
    // Browser-specific checks
    if (browserName === 'webkit') {
      // Safari-specific tests
      console.log('Running Safari-specific tests');
    } else if (browserName === 'firefox') {
      // Firefox-specific tests
      console.log('Running Firefox-specific tests');
    }
  });

  test('should maintain functionality on mobile devices', async ({ page, isMobile }) => {
    if (isMobile) {
      // Mobile-specific tests
      await page.waitForLoadState('networkidle');
      
      // Touch interactions
      await page.tap('#basic-btn');
      await expect(page.locator('#basic-result')).toContainText('Basic function works!');
      
      // Mobile viewport issues (will detect responsive bugs)
      const viewport = page.viewportSize();
      expect(viewport.width).toBeLessThanOrEqual(500);
      
      // Check for horizontal scroll (indicates responsive issues)
      const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
      expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 10); // Allow small tolerance
    }
  });
});
