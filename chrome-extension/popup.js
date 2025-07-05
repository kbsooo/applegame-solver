document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById('toggleBtn');
    const status = document.getElementById('status');
    
    console.log('Popup loaded');
    
    // Check current state
    chrome.storage.local.get(['overlayEnabled'], function(result) {
        console.log('Current state:', result);
        const enabled = result.overlayEnabled || false;
        updateUI(enabled);
    });
    
    toggleBtn.addEventListener('click', function() {
        console.log('Button clicked');
        
        chrome.storage.local.get(['overlayEnabled'], function(result) {
            const enabled = result.overlayEnabled || false;
            const newState = !enabled;
            
            console.log('Toggling from', enabled, 'to', newState);
            
            chrome.storage.local.set({ overlayEnabled: newState }, function() {
                console.log('State saved');
                updateUI(newState);
                
                // Send message to content script
                chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                    if (tabs[0]) {
                        console.log('Sending message to tab:', tabs[0].id);
                        chrome.tabs.sendMessage(tabs[0].id, {
                            action: 'test',
                            enabled: newState
                        }, function(response) {
                            if (chrome.runtime.lastError) {
                                console.log('Message error:', chrome.runtime.lastError);
                            } else {
                                console.log('Message sent successfully:', response);
                            }
                        });
                    }
                });
            });
        });
    });
    
    function updateUI(enabled) {
        console.log('Updating UI with enabled:', enabled);
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