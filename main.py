import os
import random
import numpy as np
import argparse
from entities import Course, Room, TimeSlot
from solver import exam_scheduling_main
from analytics import analysis
from reports import generate_manuel_fac_

def parse_arguments():
    parser = argparse.ArgumentParser(description="Campus Exam Scheduling System")
    parser.add_argument('--midterm', action='store_true', help='Run scheduling for Midterms (Default: Finals)')
    parser.add_argument('--demo', action='store_true', help='Run in Demo Mode (Uses cached solution for speed)')
    parser.add_argument('--timeout', type=int, default=600, help='Solver timeout in seconds (Default: 600)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--analyze', type=int, help='Skip solving and analyze a specific experiment ID')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    runs_path = "./runs"
    os.makedirs(runs_path, exist_ok=True)

    if args.analyze is not None:
        experiment = args.analyze
    else:
        experiment = len(os.listdir(runs_path))

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    is_midterm = args.midterm
    is_demo = args.demo
    timeout = args.timeout

    num_days = 10 if is_midterm else 8
    slots_per_day = 9

    print(f"--- CONFIGURATION ---")
    print(f"Exam Type: {'Midterm' if is_midterm else 'Final'}")
    print(f"Mode: {'DEMO (Warm Start)' if is_demo else 'OPTIMIZATION (Cold Start)'}")
    print(f"Timeout: {timeout} seconds")
    print(f"Experiment ID: {experiment}")
    print(f"---------------------")

    course_xlsx = "./data/course_data_B24.xlsx"
    room_xlsx = "./data/room_data.xlsx"

    # Initialize Data
    Course.read_courses(course_xlsx, is_midterm)
    Room.read_classroom_data(room_xlsx, num_days, slots_per_day, is_midterm)

    off_by_day = [[4] for _ in range(num_days)]
    if num_days > 4:
        off_by_day[4] = off_by_day[4] + [5]
    if num_days >= 10:
        off_by_day[9] = off_by_day[9] + [5]
    if num_days > 0 and not is_midterm:
        off_by_day[0] = off_by_day[0] + [5, 6]  # simulations of 5i exams

    TimeSlot.generate_week(num_days, slots_per_day, off_by_day)

    if args.analyze is None:
        exam_scheduling_main(experiment, is_midterm, num_days, slots_per_day, timeout, runs_path, is_demo)
        analysis(experiment, is_midterm, num_days)
    else:

        analysis(experiment, is_midterm, num_days)