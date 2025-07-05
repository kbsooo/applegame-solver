// Test content script for debugging
console.log('Apple Game Solver: Test content script loaded');
console.log('Current URL:', window.location.href);
console.log('Document ready state:', document.readyState);

// Test if we can find any canvas elements
const canvases = document.querySelectorAll('canvas');
console.log('Found canvases:', canvases.length);
canvases.forEach((canvas, index) => {
    console.log(`Canvas ${index}:`, canvas.id, canvas.className, canvas.width, canvas.height);
});

// Test if we can create a simple overlay
function createTestOverlay() {
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '10px';
    overlay.style.left = '10px';
    overlay.style.background = 'rgba(255, 0, 0, 0.8)';
    overlay.style.color = 'white';
    overlay.style.padding = '10px';
    overlay.style.borderRadius = '5px';
    overlay.style.zIndex = '10000';
    overlay.style.fontSize = '14px';
    overlay.textContent = 'Apple Game Solver 테스트 오버레이';
    document.body.appendChild(overlay);
    
    console.log('Test overlay created');
    
    // Remove after 3 seconds
    setTimeout(() => {
        overlay.remove();
        console.log('Test overlay removed');
    }, 3000);
}

// Wait for page to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createTestOverlay);
} else {
    createTestOverlay();
}

// Listen for messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Test content script received message:', request);
    if (request.action === 'test') {
        createTestOverlay();
        sendResponse({ success: true, message: 'Test overlay created' });
    }
    return true;
});