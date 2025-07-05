// Apple Game Solver Content Script
class AppleGameOverlay {
    constructor() {
        this.solver = new AppleGameSolver();
        this.overlayEnabled = false;
        this.overlay = null;
        this.gameCanvas = null;
        this.gameState = null;
        this.updateInterval = null;
        
        this.init();
    }

    init() {
        // Wait for page to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        console.log('Apple Game Solver: Setting up...');
        console.log('Current URL:', window.location.href);
        
        // Find the game canvas
        this.gameCanvas = document.getElementById('canvas');
        if (!this.gameCanvas) {
            console.log('Apple Game Solver: Canvas not found, searching for alternative selectors...');
            
            // Try alternative selectors
            this.gameCanvas = document.querySelector('canvas') || 
                            document.querySelector('#game-canvas') ||
                            document.querySelector('.game-canvas');
            
            if (!this.gameCanvas) {
                console.log('Apple Game Solver: No canvas found, retrying in 2 seconds...');
                setTimeout(() => this.setup(), 2000);
                return;
            } else {
                console.log('Apple Game Solver: Found canvas with alternative selector');
            }
        } else {
            console.log('Apple Game Solver: Canvas found with ID "canvas"');
        }

        // Create overlay
        this.createOverlay();
        
        // Check initial state
        chrome.storage.local.get(['overlayEnabled'], (result) => {
            this.overlayEnabled = result.overlayEnabled || false;
            if (this.overlayEnabled) {
                this.startOverlay();
            }
        });

        // Listen for messages from popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            console.log('Content script received message:', request);
            if (request.action === 'toggleOverlay') {
                this.overlayEnabled = request.enabled;
                if (this.overlayEnabled) {
                    console.log('Starting overlay');
                    this.startOverlay();
                } else {
                    console.log('Stopping overlay');
                    this.stopOverlay();
                }
                sendResponse({ success: true });
            }
            return true; // Keep message channel open
        });

        console.log('Apple Game Solver: Initialized');
    }

    createOverlay() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'apple-solver-overlay';
        this.overlay.style.display = 'none';
        document.body.appendChild(this.overlay);
    }

    startOverlay() {
        if (!this.overlay) return;
        
        this.overlay.style.display = 'block';
        this.addStatusIndicator();
        
        // Start monitoring game state
        this.updateInterval = setInterval(() => {
            this.updateSuggestions();
        }, 1000);
        
        console.log('Apple Game Solver: Overlay started');
    }

    stopOverlay() {
        if (!this.overlay) return;
        
        this.overlay.style.display = 'none';
        
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        
        console.log('Apple Game Solver: Overlay stopped');
    }

    addStatusIndicator() {
        const status = document.createElement('div');
        status.className = 'apple-solver-status';
        status.textContent = 'Apple Game Solver: 활성화됨';
        this.overlay.appendChild(status);
    }

    // This is a mock implementation - in a real scenario, you'd need to:
    // 1. Extract actual game state from the canvas
    // 2. Convert canvas coordinates to game board positions
    // 3. Detect when the game state changes
    extractGameStateFromCanvas() {
        // Mock implementation - replace with actual canvas analysis
        const board = [];
        const cleared = [];
        
        for (let i = 0; i < 10; i++) {
            const row = [];
            const clearedRow = [];
            for (let j = 0; j < 17; j++) {
                row.push(Math.floor(Math.random() * 9) + 1);
                clearedRow.push(false);
            }
            board.push(row);
            cleared.push(clearedRow);
        }
        
        return { board, cleared };
    }

    updateSuggestions() {
        if (!this.overlayEnabled || !this.gameCanvas) return;
        
        try {
            // Extract current game state
            const gameState = this.extractGameStateFromCanvas();
            
            // Get best moves
            const bestMoves = this.solver.getBestMoves(gameState.board, gameState.cleared, 3);
            
            // Clear previous suggestions
            this.clearSuggestions();
            
            // Add new suggestions
            bestMoves.forEach((move, index) => {
                this.addSuggestion(move, index === 0);
            });
            
        } catch (error) {
            console.error('Apple Game Solver: Error updating suggestions:', error);
        }
    }

    clearSuggestions() {
        const suggestions = this.overlay.querySelectorAll('.apple-solver-suggestion, .apple-solver-highlight');
        suggestions.forEach(suggestion => suggestion.remove());
    }

    addSuggestion(move, isBest = false) {
        const canvasRect = this.gameCanvas.getBoundingClientRect();
        
        // Calculate approximate position on canvas
        // This is a mock calculation - you'd need to map game coordinates to canvas pixels
        const cellWidth = canvasRect.width / 17;
        const cellHeight = canvasRect.height / 10;
        
        // Create highlight for the move area
        const highlight = document.createElement('div');
        highlight.className = `apple-solver-highlight ${isBest ? 'best' : ''}`;
        
        // Calculate bounding box of the move
        const rows = move.positions.map(pos => pos[0]);
        const cols = move.positions.map(pos => pos[1]);
        const minRow = Math.min(...rows);
        const maxRow = Math.max(...rows);
        const minCol = Math.min(...cols);
        const maxCol = Math.max(...cols);
        
        const left = canvasRect.left + (minCol * cellWidth);
        const top = canvasRect.top + (minRow * cellHeight);
        const width = (maxCol - minCol + 1) * cellWidth;
        const height = (maxRow - minRow + 1) * cellHeight;
        
        highlight.style.left = `${left}px`;
        highlight.style.top = `${top}px`;
        highlight.style.width = `${width}px`;
        highlight.style.height = `${height}px`;
        
        this.overlay.appendChild(highlight);
        
        // Add suggestion label
        const suggestion = document.createElement('div');
        suggestion.className = `apple-solver-suggestion ${isBest ? 'best' : ''}`;
        suggestion.textContent = `합계: 10 (${move.positions.length}개)`;
        
        suggestion.style.left = `${left}px`;
        suggestion.style.top = `${top - 25}px`;
        
        this.overlay.appendChild(suggestion);
    }
}

// Load the model and start the overlay
const script = document.createElement('script');
script.src = chrome.runtime.getURL('model.js');
script.onload = () => {
    new AppleGameOverlay();
};
document.head.appendChild(script);