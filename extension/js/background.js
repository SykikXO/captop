/**
 * VTOP Captcha Solver - Background Script
 * Handles network requests to bypass Content Security Policy (CSP) of the host page.
 */

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'solve') {
        handleSolve(request.image).then(sendResponse);
        return true; // Keep message channel open for async response
    } else if (request.action === 'report') {
        handleReport(request.image, request.prediction).then(sendResponse);
        return true;
    }
});

async function handleSolve(image) {
    try {
        const apiUrl = await getApiUrl();
        const response = await fetch(`${apiUrl}/solve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image })
        });

        if (!response.ok) throw new Error('API error');
        const data = await response.json();
        return { success: true, text: data.text };
    } catch (error) {
        console.error('[Background] Solve error:', error);
        return { success: false, error: error.message };
    }
}

async function handleReport(image, prediction) {
    try {
        const apiUrl = await getApiUrl();
        await fetch(`${apiUrl}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image, prediction })
        });
        return { success: true };
    } catch (error) {
        console.error('[Background] Report error:', error);
        return { success: false, error: error.message };
    }
}

async function getApiUrl() {
    const stored = await chrome.storage.sync.get(['apiUrl']);
    return stored.apiUrl || 'https://captop-proxy.sykik.workers.dev';
}
