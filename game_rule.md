# Context Document: The "Apple Game" (사과게임, フルーツボックス)

## 1. Executive Summary

This document provides a comprehensive overview of the puzzle game "Fruit Box" (フルーツボックス), commonly known as the "Apple Game" (사과게임). It is a time-attack puzzle game where the objective is to clear a grid of numbered apples by selecting groups whose numbers sum to exactly 10. The core mechanic, and primary source of strategy, is that clearing apples leaves permanent empty spaces on the board. The player's selection tool can **ignore these empty spaces**, allowing for the selection and combination of **non-adjacent apples**.

## 2. Game Overview

- **Official Name:** フルーツボックス (Fruit Box)
- **Developer:** ゲーム菜園 (Game-SaiEn)
- **Common Name:** 사과게임 (Apple Game)
- **Platform:** Originally Adobe Flash, now HTML5 (Web-based)
- **Genre:** Puzzle
- **First Released:** October 15, 2007

## 3. Core Gameplay Mechanics

### Objective
The player's goal is to score the maximum number of points within a **2-minute (120-second)** time limit by clearing apples from the board.

### Game Board
- The game starts with a **17x10 grid** containing **170 apples**.
- Each apple is marked with a single digit from **1 to 9**.
- The distribution of these numbers is **completely random** for each new game.

### Controls & Action (The Corrected Mechanic)
1.  The player uses the mouse to **click and drag**, creating a rectangular selection box.
2.  The selection box can be dragged over any area of the grid, including areas with empty spaces.
3.  The game calculates the sum based **only on the apples** within the selection box, completely **ignoring any empty spaces** it covers.
4.  If the sum of the selected, potentially non-adjacent apples is **exactly 10**, the box turns red.
5.  Releasing the mouse button will **clear the selected apples**, leaving **permanent empty spaces**. The board is not refilled; it depletes as apples are cleared.

**Example:** As shown in the provided screenshot, a player can draw a box that covers a '6', an empty space, and a '4'. The game ignores the space and calculates `6 + 4 = 10`, allowing the player to clear these two non-adjacent apples.

## 4. Scoring

- **Points:** Players earn **1 point for each apple cleared**.
- **Maximum Score:** Since there are 170 apples, the theoretical maximum score is **170**.
- **Perfect Clear:** If a player clears all 170 apples before the 2-minute timer expires, the game ends and displays the total time taken.

## 5. Strategic Analysis & Combinations

The game's strategy is not about managing falling blocks, but about spatial clearing to create "lines of sight" between numbers.

### The "Impossible Board" Problem
This issue remains critical. If the initial random generation provides more '9's than '1's (or a similar imbalance), the board is **mathematically impossible to clear**, regardless of skill. High-level play still involves resetting for a favorable number distribution.

### "Line-of-Sight" Clearing Strategy
The true skill of the game lies in what to clear and when, in order to enable future moves.
- **Path Creation:** The primary goal is to remove "obstructing" apples to open up paths between numbers that add up to 10. A player might clear a low-value pair simply to remove the apples sitting physically between a distant '8' and '2'.
- **Board Depletion:** As the board empties, it becomes progressively easier to connect numbers across large distances. The strategy evolves from clearing dense local clusters to sniping widely separated pairs.
- **Strategic Sacrifices:** Sometimes it's necessary to clear a combination that isn't ideal in order to enable a more critical combination. The entire board is a static puzzle that the player strategically carves out.

### Practical Combinations to Memorize (4 or fewer apples)
This list remains the foundation for quick calculation, regardless of the board mechanic.

- **Containing 9 (1 combo):** `9-1`
- **Containing 8 (2 combos):** `8-2`, `8-1-1`
- **Containing 7 (3 combos):** `7-3`, `7-2-1`, `7-1-1-1`
- **Containing 6 (4 combos):** `6-4`, `6-3-1`, `6-2-2`, `6-2-1-1`
- **Containing 5 (5 combos):** `5-5`, `5-4-1`, `5-3-2`, `5-3-1-1`, `5-2-2-1`
- **Containing 4 (5 combos):** `4-4-2`, `4-3-3`, `4-4-1-1`, `4-3-2-1`, `4-2-2-2`
- **Containing 3 (2 combos):** `3-3-3-1`, `3-3-2-2`

## 6. Context and Modern Variants

(This section remains unchanged as it pertains to the game's history and ecosystem, which are unaffected by the core mechanic clarification.)