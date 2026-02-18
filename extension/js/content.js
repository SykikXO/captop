/**
 * VTOP Captcha Solver - Content Script
 * Automatically detects and solves captchas on the login page.
 */


// Track current captcha state
let currentCaptchaBase64 = null;
let currentPrediction = null;

/**
 * Initialize the solver when the page loads
 */
async function init() {

  const captchaImg = document.querySelector('#captchaBlock img');
  const captchaInput = document.getElementById('captchaStr');

  if (!captchaImg || !captchaInput) {
    console.log('[CaptchaSolver] Login form not found, retrying...');
    setTimeout(init, 500);
    return;
  }

  console.log('[CaptchaSolver] Initialized');

  // Solve on initial load
  solveCaptcha();

  // Watch for captcha refresh
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
        solveCaptcha();
      }
    }
  });

  observer.observe(captchaImg, { attributes: true });

  // Watch for login failure to report
  watchLoginResult();
}

/**
 * Main solve function
 */
async function solveCaptcha() {
  const captchaImg = document.querySelector('#captchaBlock img');
  const captchaInput = document.getElementById('captchaStr');

  if (!captchaImg || !captchaInput) return;

  // Extract base64 from src (remove data:image/jpeg;base64, prefix)
  const src = captchaImg.src;
  if (!src.startsWith('data:image')) {
    console.log('[CaptchaSolver] Invalid captcha src');
    return;
  }

  currentCaptchaBase64 = src.split(',')[1];

  // Show loading bar
  showLoadingBar(captchaInput);

  try {
    // Send to background script to bypass CSP
    chrome.runtime.sendMessage({
      action: 'solve',
      image: currentCaptchaBase64
    }, (response) => {
      if (!response || !response.success) {
        throw new Error(response?.error || 'Background script error');
      }

      currentPrediction = response.text;
      hideLoadingBar(captchaInput, response.text);
      console.log(`[CaptchaSolver] Solved: ${response.text}`);
    });

  } catch (error) {
    console.error('[CaptchaSolver] Solve Error:', error);
    hideLoadingBar(captchaInput, '');
  }
}

/**
 * Replace input with Windows 7-style loading bar
 */
function showLoadingBar(input) {
  input.value = '';
  input.placeholder = '';
  input.classList.add('vtop-loading');
  input.readOnly = true;
}

/**
 * Restore input and fill with solved text
 */
function hideLoadingBar(input, text) {
  input.classList.remove('vtop-loading');
  input.classList.add('vtop-solved');
  input.readOnly = false;
  input.value = text;

  // Remove solved class after animation
  setTimeout(() => {
    input.classList.remove('vtop-solved');
  }, 500);
}

/**
 * Watch for login failures to report bad predictions
 */
function watchLoginResult() {
  // Watch for error messages appearing
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const text = node.textContent?.toLowerCase() || '';
          if (text.includes('invalid') || text.includes('captcha') || text.includes('wrong')) {
            reportFailure();
          }
        }
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

/**
 * Report failed captcha to server for retraining
 */
async function reportFailure() {
  if (!currentCaptchaBase64 || !currentPrediction) return;

  console.log('[CaptchaSolver] Reporting failed prediction');

  try {
    chrome.runtime.sendMessage({
      action: 'report',
      image: currentCaptchaBase64,
      prediction: currentPrediction
    });
  } catch (error) {
    console.error('[CaptchaSolver] Report error:', error);
  }

  // Clear to prevent duplicate reports
  currentCaptchaBase64 = null;
  currentPrediction = null;
}

// Start
init();
