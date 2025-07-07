document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById('toggleBtn');
    const testBtn = document.getElementById('testBtn');
    const status = document.getElementById('status');
    
    console.log('Popup loaded');
    
    // Check current state
    chrome.storage.local.get(['overlayEnabled'], function(result) {
        console.log('Current state:', result);
        const enabled = result.overlayEnabled || false;
        updateUI(enabled);
    });
    
    // Test button - simpler direct test
    testBtn.addEventListener('click', function() {
        console.log('Test button clicked');
        
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                console.log('Sending test message to tab:', tabs[0].id);
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'test'
                }, function(response) {
                    if (chrome.runtime.lastError) {
                        console.log('Test message error:', chrome.runtime.lastError);
                    } else {
                        console.log('Test message sent successfully:', response);
                    }
                });
            } else {
                console.log('No active tab found');
            }
        });
    });
    
    toggleBtn.addEventListener('click', function() {
        console.log('Toggle button clicked');
        
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
                        console.log('Sending toggle message to tab:', tabs[0].id);
                        chrome.tabs.sendMessage(tabs[0].id, {
                            action: 'toggleOverlay',
                            enabled: newState
                        }, function(response) {
                            if (chrome.runtime.lastError) {
                                console.log('Toggle message error:', chrome.runtime.lastError);
                            } else {
                                console.log('Toggle message sent successfully:', response);
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