// JavaScript implementation of the Apple Game solver
class AppleGameSolver {
    constructor() {
        this.rows = 10;
        this.cols = 17;
    }

    // Extract game board state from canvas
    extractGameState() {
        // This would need to be implemented based on the actual game structure
        // For now, return a mock board for testing
        const board = [];
        for (let i = 0; i < this.rows; i++) {
            const row = [];
            for (let j = 0; j < this.cols; j++) {
                row.push(Math.floor(Math.random() * 9) + 1);
            }
            board.push(row);
        }
        return board;
    }

    // Check if positions form a valid rectangle
    canSelectRectangle(positions) {
        if (positions.length < 2) return false;
        
        const rows = positions.map(pos => pos[0]);
        const cols = positions.map(pos => pos[1]);
        
        const minRow = Math.min(...rows);
        const maxRow = Math.max(...rows);
        const minCol = Math.min(...cols);
        const maxCol = Math.max(...cols);
        
        // Check if all positions within the rectangle are included
        for (let r = minRow; r <= maxRow; r++) {
            for (let c = minCol; c <= maxCol; c++) {
                if (!positions.some(pos => pos[0] === r && pos[1] === c)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Find all valid combinations that sum to 10
    findValidCombinations(board, cleared) {
        const validMoves = [];
        const availablePositions = [];
        
        // Get all non-cleared positions
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                if (!cleared[r][c]) {
                    availablePositions.push([r, c]);
                }
            }
        }
        
        // Try all possible rectangular combinations
        for (let i = 0; i < availablePositions.length; i++) {
            for (let j = i + 1; j < Math.min(i + 20, availablePositions.length); j++) {
                const subset = availablePositions.slice(i, j + 1);
                if (this.canSelectRectangle(subset)) {
                    const values = subset.map(pos => board[pos[0]][pos[1]]);
                    if (values.reduce((a, b) => a + b, 0) === 10) {
                        validMoves.push({
                            positions: subset,
                            values: values,
                            score: values.length
                        });
                    }
                }
            }
        }
        
        return validMoves;
    }

    // Simple heuristic scoring (replace with actual ML model logic)
    scoreMove(move, board) {
        // Prioritize moves with more cells (higher score)
        let score = move.score * 10;
        
        // Bonus for moves that clear corners or edges
        for (const pos of move.positions) {
            const [r, c] = pos;
            if (r === 0 || r === this.rows - 1 || c === 0 || c === this.cols - 1) {
                score += 5;
            }
            if ((r === 0 || r === this.rows - 1) && (c === 0 || c === this.cols - 1)) {
                score += 10; // Corner bonus
            }
        }
        
        return score;
    }

    // Get best moves
    getBestMoves(board, cleared, topN = 3) {
        const validMoves = this.findValidCombinations(board, cleared);
        
        // Score and sort moves
        const scoredMoves = validMoves.map(move => ({
            ...move,
            aiScore: this.scoreMove(move, board)
        }));
        
        scoredMoves.sort((a, b) => b.aiScore - a.aiScore);
        
        return scoredMoves.slice(0, topN);
    }
}

// Export for use in content script
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AppleGameSolver;
} else {
    window.AppleGameSolver = AppleGameSolver;
}