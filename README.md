# Campus Exam Scheduling System

An automated constraint programming solution for university exam timetabling. This system uses Google OR-Tools to generate conflict-free schedules for Midterm and Final exams, optimizing for room capacity, student overlap, and instructor availability.

## Installation

0. **Get the Code (Demo Branch)**:
   ```bash
   git clone https://github.com/ErcxCS/Campus-Scheduling.git
   cd Campus-Scheduling
   git checkout demo
   ```

1. **Prerequisites**: Ensure you have Python 3.8+ installed.
2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## How to Run

The system is controlled via the command line using `main.py`.

### Basic Syntax

```bash
python main.py [arguments]
```

### Arguments Explanation

| Argument | Description |
|---|---|
| *(No Argument)* | Runs a standard Final Exam optimization (8 days) using the default B24 dataset. |
| `--midterm` | Switches the mode to Midterm Exams (10 days). Default is Finals. |
| `--demo` | Demo Mode: Uses a pre-calculated solution ("warm start") to finish in seconds. Useful for presentations. |
| `--G25` | Switches the input dataset to G25 (Fall 2025 data). Note: Post-run analysis is disabled for this dataset. |
| `--timeout N` | Sets the solver time limit in seconds (Default: 600). |
| `--seed N` | Sets a random seed for reproducibility. |
| `--num_days N` | Set the number of days, by default if for final 8, if midterm 10. |
| `--analyze ID` | Skips the solver and runs the analysis suite on a specific Experiment ID (e.g., `--analyze 5`). |

## Important Note on Analysis

The Analysis Module (graphs, heatmaps, and comparison reports) is strictly calibrated for the B24 (Spring 2024) dataset.

- **B24 Runs**: Analysis runs automatically after the solver finishes.
- **G25 Runs**: Analysis is disabled automatically. The code will generate the schedule Excel files but will skip the statistical reports.

## Usage Examples

### 1. Standard Final Exam Run (B24)

Runs optimization for 10 minutes (default) and generates a schedule.

```bash
python main.py
```

### 2. Final Demo 

Uses cached hints to generate a valid midterm schedule in ~10-30 seconds.

```bash
python main.py --demo
```

### 3. Midterm Demo

Uses cached hints to generate a valid midterm schedule in ~10-30 seconds.

```bash
python main.py --midterm --demo
```

### 4. Run with New G25 Data

Generates a schedule using the G25 course data (Analysis skipped).

```bash
python main.py --G25
```

### 5. Re-Analyze an Old Experiment

Regenerates the graphs and reports for experiment #3 (Finals) without running the solver again.

```bash
python main.py --analyze 3
```

## Project Structure

- `main.py`: The entry point. Handles argument parsing and orchestrates the flow.
- `entities.py`: Data models (Course, Room, Department, TimeSlot).
- `solver.py`: The core OR-Tools CP-SAT logic and constraints.
- `reports.py`: Excel formatting and file generation logic.
- `analytics.py`: Statistical analysis and plotting functions.
- `utils.py`: Low-level file I/O and helper functions.
