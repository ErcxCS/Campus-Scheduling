import random
import argparse
import numpy as np

from src.config import Config, FACULTIES, SEMESTERS, RUNS_DIR
from src.entities import Course, Room, TimeSlot
from src.solver import exam_scheduling_main
from src.analytics import analysis


def parse_arguments():
    parser = argparse.ArgumentParser(description="Campus Exam Scheduling System")
    parser.add_argument('--faculty', choices=FACULTIES, default='eng',
                        help='Faculty to schedule (default: eng)')
    parser.add_argument('--semester', choices=SEMESTERS, default='B24',
                        help='Semester dataset (default: B24)')
    parser.add_argument('--midterm', action='store_true',
                        help='Run scheduling for Midterms (default: Finals)')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: use cached solution for fast warm start')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Solver timeout in seconds (default: 600)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-days', type=int, default=None,
                        help='Override schedule duration in days')
    parser.add_argument('--analyze', type=int, default=None,
                        help='Skip solving and re-analyze a specific experiment ID')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom experiment name (default: auto-increment)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = Config.from_args(args)

    RUNS_DIR.mkdir(exist_ok=True)
    cfg.validate()

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Load data
    Course.read_courses(str(cfg.course_path), cfg.is_midterm)
    Room.read_classroom_data(str(cfg.room_path), cfg.num_days, cfg.slots_per_day, cfg.is_midterm)

    print("--- CONFIGURATION ---")
    print(f"Faculty:   {cfg.faculty}")
    print(f"Semester:  {cfg.semester}")
    print(f"Exam Type: {'Midterm' if cfg.is_midterm else 'Final'}")
    print(f"Mode:      {'DEMO (Warm Start)' if cfg.is_demo else 'OPTIMIZATION (Cold Start)'}")
    print(f"Duration:  {cfg.num_days} days")
    print(f"Timeout:   {cfg.timeout} seconds")
    print(f"Experiment: {cfg.experiment_id}")
    print(f"Courses:   {len(Course.course_list)}")
    print(f"Rooms:     {len(Room.room_list)}")
    print("---------------------")

    # Generate time slots with off-times
    off_by_day = [[4] for _ in range(cfg.num_days)]
    if cfg.num_days > 4:
        off_by_day[4] = off_by_day[4] + [5]
    if cfg.num_days > 9:
        off_by_day[9] = off_by_day[9] + [5]

    TimeSlot.generate_week(cfg.num_days, cfg.slots_per_day, off_by_day)

    if args.analyze is None:
        exam_scheduling_main(cfg)
        if cfg.has_reference:
            analysis(cfg)
        else:
            print(f"Skipping analysis (no reference data for {cfg.faculty}/{cfg.semester}).")
    else:
        if cfg.has_reference:
            analysis(cfg)
        else:
            print(f"Cannot analyze: no reference data for {cfg.faculty}/{cfg.semester}.")


if __name__ == "__main__":
    main()
