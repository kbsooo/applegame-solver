document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById('toggleBtn');
    const status = document.getElementById('status');
    
    // Check current state
    chrome.storage.local.get(['overlayEnabled'], function(result) {
        const enabled = result.overlayEnabled || false;
        updateUI(enabled);
    });
    
    toggleBtn.addEventListener('click', function() {
        chrome.storage.local.get(['overlayEnabled'], function(result) {
            const enabled = result.overlayEnabled || false;
            const newState = !enabled;
            
            chrome.storage.local.set({ overlayEnabled: newState }, function() {
                updateUI(newState);
                
                // Send message to content script
                chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                    chrome.tabs.sendMessage(tabs[0].id, {
                        action: 'toggleOverlay',
                        enabled: newState
                    });
                });
            });
        });
    });
    
    function updateUI(enabled) {
        if (enabled) {
            status.textContent = '오버레이 활성화됨';
            status.className = 'status active';
            toggleBtn.textContent = '오버레이 비활성화';
        } else {
            status.textContent = '오버레이 비활성화됨';
            status.className = 'status inactive';
            toggleBtn.textContent = '오버레이 활성화';
        }
    }
});