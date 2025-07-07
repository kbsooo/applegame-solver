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
            if (request.action === 'test') {
                console.log('Test action received');
                this.testGameAnalysis();
                sendResponse({ success: true, message: 'Test completed' });
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

    // Extract game state from canvas using pixel analysis
    extractGameStateFromCanvas() {
        if (!this.gameCanvas) return null;
        
        const ctx = this.gameCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, this.gameCanvas.width, this.gameCanvas.height);
        
        // Analyze canvas to find game board
        const board = [];
        const cleared = [];
        
        // Calculate cell dimensions based on canvas size
        const cellWidth = this.gameCanvas.width / 17;
        const cellHeight = this.gameCanvas.height / 10;
        
        for (let row = 0; row < 10; row++) {
            const boardRow = [];
            const clearedRow = [];
            
            for (let col = 0; col < 17; col++) {
                // Calculate center position of each cell
                const centerX = Math.floor(col * cellWidth + cellWidth / 2);
                const centerY = Math.floor(row * cellHeight + cellHeight / 2);
                
                // Sample multiple points around the center to detect numbers
                const value = this.detectNumberInCell(imageData, centerX, centerY, cellWidth, cellHeight);
                const isEmpty = this.isCellEmpty(imageData, centerX, centerY, cellWidth, cellHeight);
                
                boardRow.push(value || Math.floor(Math.random() * 9) + 1); // Fallback to random
                clearedRow.push(isEmpty);
            }
            
            board.push(boardRow);
            cleared.push(clearedRow);
        }
        
        return { board, cleared };
    }
    
    // Detect if a cell is empty (cleared)
    isCellEmpty(imageData, centerX, centerY, cellWidth, cellHeight) {
        const samplePoints = [
            [centerX, centerY],
            [centerX - cellWidth/4, centerY],
            [centerX + cellWidth/4, centerY],
            [centerX, centerY - cellHeight/4],
            [centerX, centerY + cellHeight/4]
        ];
        
        let emptyCount = 0;
        for (const [x, y] of samplePoints) {
            if (x >= 0 && x < imageData.width && y >= 0 && y < imageData.height) {
                const index = (Math.floor(y) * imageData.width + Math.floor(x)) * 4;
                const r = imageData.data[index];
                const g = imageData.data[index + 1];
                const b = imageData.data[index + 2];
                const a = imageData.data[index + 3];
                
                // Check if pixel is transparent or very light (indicating empty cell)
                if (a < 50 || (r > 200 && g > 200 && b > 200)) {
                    emptyCount++;
                }
            }
        }
        
        return emptyCount > samplePoints.length / 2;
    }
    
    // Detect number in a cell (simplified implementation)
    detectNumberInCell(imageData, centerX, centerY, cellWidth, cellHeight) {
        // This is a simplified approach - in reality, you'd need OCR or pattern matching
        // For now, return a random number between 1-9
        return Math.floor(Math.random() * 9) + 1;
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
    
    // Test method to analyze game and show debug info
    testGameAnalysis() {
        console.log('Testing game analysis...');
        
        if (!this.gameCanvas) {
            console.log('No canvas found');
            return;
        }
        
        // Extract game state
        const gameState = this.extractGameStateFromCanvas();
        if (!gameState) {
            console.log('Failed to extract game state');
            return;
        }
        
        console.log('Game board extracted:', gameState.board);
        console.log('Cleared cells:', gameState.cleared);
        
        // Get suggestions
        const bestMoves = this.solver.getBestMoves(gameState.board, gameState.cleared, 3);
        console.log('Best moves found:', bestMoves);
        
        // Show visual overlay with suggestions
        this.clearSuggestions();
        bestMoves.forEach((move, index) => {
            console.log(`Move ${index + 1}:`, move.positions, 'Score:', move.aiScore);
            this.addSuggestion(move, index === 0);
        });
        
        // Create test overlay to show analysis
        const testOverlay = document.createElement('div');
        testOverlay.style.position = 'fixed';
        testOverlay.style.top = '10px';
        testOverlay.style.right = '10px';
        testOverlay.style.background = 'rgba(0, 0, 0, 0.9)';
        testOverlay.style.color = 'white';
        testOverlay.style.padding = '10px';
        testOverlay.style.borderRadius = '5px';
        testOverlay.style.zIndex = '10000';
        testOverlay.style.fontSize = '12px';
        testOverlay.style.maxWidth = '300px';
        testOverlay.innerHTML = `
            <div><strong>Game Analysis:</strong></div>
            <div>Canvas: ${this.gameCanvas.width}x${this.gameCanvas.height}</div>
            <div>Best moves: ${bestMoves.length}</div>
            <div>Board sample: ${gameState.board[0].slice(0, 5).join(',')}</div>
            <div>Cleared sample: ${gameState.cleared[0].slice(0, 5).join(',')}</div>
        `;
        document.body.appendChild(testOverlay);
        
        // Remove after 5 seconds
        setTimeout(() => {
            testOverlay.remove();
        }, 5000);
    }
}

// Initialize the overlay
new AppleGameOverlay();