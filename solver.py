from xml.parsers.expat import model

from ortools.sat.python import cp_model
import os
from entities import Course, Room, Department, TimeSlot
from utils import get_off_chunks
from reports import build_timetable, department_exam_schedule, faculty_exam_schedule, excelify
from analytics import mission_report
import json


def save_demo_data(path, start_vars, in_room_vars, solver):
    """Saves the solution to a JSON file."""
    data = {
        "start": {},
        "in_room": {}
    }

    for cid, var in start_vars.items():
        data["start"][str(cid)] = solver.Value(var)

    for (cid, rid), var in in_room_vars.items():
        if solver.Value(var) == 1:
            key = f"{cid},{rid}" 
            data["in_room"][key] = 1

    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"\n[DEMO] Solution saved to: {path}")


def load_demo_hints(path, model, start_vars, in_room_vars):
    """Loads JSON data and injects it as hints."""
    if not os.path.exists(path):
        print(f"\n[DEMO ERROR] File {path} not found! Cannot warm start.")
        return False

    with open(path, 'r') as f:
        data = json.load(f)

    count = 0
    for cid_str, val in data["start"].items():
        cid = int(cid_str)
        if cid in start_vars:
            model.AddHint(start_vars[cid], val)
            count += 1

    for key_str, val in data["in_room"].items():
        c_str, r_str = key_str.split(",")
        cid, rid = int(c_str), int(r_str)
        if (cid, rid) in in_room_vars:
            model.AddHint(in_room_vars[(cid, rid)], val)
    
    print(f"[DEMO] Loaded {count} hints from {path}.")
    return True


def exam_scheduling_main(experiment_no: int, is_midterm: bool, num_days: int,
                         slots_per_day: int, timeout: int = 600,
                         runs_path: str = "./runs", demo_mode: bool = False,
                         dataset_name: str = "B24"):

    horizon = num_days * slots_per_day
    model = cp_model.CpModel()

    # (1) Interval / start & end variables
    start = {}
    end = {}
    interval = {}
    if is_midterm:
        for e in Course.course_list:
            dur = e.get_duration()
            if e.year in {1, 3}:
                start[e.id] = model.NewIntVar(0, horizon // 2 - dur, f"start_e{e.id}")
                end[e.id] = model.NewIntVar(0, horizon // 2, f"end_e{e.id}")
            else:
                start[e.id] = model.NewIntVar(horizon // 2, horizon - dur, f"start_e{e.id}")
                end[e.id] = model.NewIntVar(horizon // 2, horizon, f"end_e{e.id}")
            model.Add(end[e.id] == start[e.id] + dur)
            interval[e.id] = model.NewIntervalVar(start[e.id], dur, end[e.id], f"interval_e{e.id}")
    else:
        for e in Course.course_list:
            dur = e.get_duration()
            start[e.id] = model.NewIntVar(0, horizon - dur, f"start_e{e.id}")
            end[e.id] = model.NewIntVar(0, horizon, f"end_e{e.id}")
            model.Add(end[e.id] == start[e.id] + dur)
            interval[e.id] = model.NewIntervalVar(start[e.id], dur, end[e.id], f"interval_e{e.id}")

    # (2) Seat & in_room variables
    seat = {}
    in_room = {}
    for e in Course.course_list:
        for r in Room.room_list:
            cap = r.capacity // 2 if not r.is_lab else r.capacity
            seat[(e.id, r.id)] = model.NewIntVar(0, cap, f"seat_e{e.id}_r{r.id}")
            in_room[(e.id, r.id)] = model.NewBoolVar(f"in_room_e{e.id}_r{r.id}")
            model.Add(seat[(e.id, r.id)] <= cap * in_room[(e.id, r.id)])

    use_fac = {}
    for e in Course.course_list:
        for f in Room.faculties:
            var = model.NewBoolVar(f"use_fac_e{e.id}_f{f}")
            use_fac[(e.id, f)] = var
        model.Add(sum(use_fac[(e.id, f)] for f in Room.faculties) == 1)

        for f in Room.faculties:
            for r in Room.rooms_by_fac[f]:
                model.Add(in_room[(e.id, r.id)] <= use_fac[(e.id, f)])

    # (3) Full-seat constraint
    for e in Course.course_list:
        model.Add(sum(seat[(e.id, r.id)] for r in Room.room_list) == e.n_students)

    # (4) Optional intervals + room-level capacity
    SMALL_EXAM_THRESHOLD = Room.min_capacity // 2
    
    opt_int_per_room = {r.id: [] for r in Room.room_list}
    for e in Course.course_list:
        for r in Room.room_list:
            opt = model.NewOptionalIntervalVar(
                start[e.id],
                e.get_duration(),
                end[e.id],
                in_room[(e.id, r.id)],
                f"opt_e{e.id}_r{r.id}"
            )
            opt_int_per_room[r.id].append(opt)

    for r in Room.room_list:
        seat_demands = [seat[(e.id, r.id)] for e in Course.course_list]
        cap = r.capacity if r.is_lab else r.capacity // 2
        model.AddCumulative(
            intervals=opt_int_per_room[r.id],
            demands=seat_demands,
            capacity=cap
        )
        # Limit simultaneous exams
        model.AddCumulative(
            intervals=opt_int_per_room[r.id],
            demands=[1] * len(Course.course_list),
            capacity=2
        )
        # NEW: forbid two NON-SMALL exams overlapping in the same room
        # big demand = 1, small demand = 0
        big_demands = [
            1 if e.n_students > SMALL_EXAM_THRESHOLD else 0
            for e in Course.course_list
        ]
        model.AddCumulative(
            intervals=opt_int_per_room[r.id],
            demands=big_demands,
            capacity=1
        )

    # Not needed when simultaneous exam capacity is 1
    """ # (5) Exams in the same room that overlap must start at the same time
    for r in Room.room_list:
        for i in range(len(Course.course_list)):
            for j in range(i+1, len(Course.course_list)):
                e = Course.course_list[i]
                f = Course.course_list[j]
                dur_e = e.get_duration()
                dur_f = f.get_duration()

                # 1) b1: e finishes ≤ f starts
                b1 = model.NewBoolVar(f"e{e.id}_before_f{f.id}")
                model.Add(start[e.id] + dur_e <= start[f.id]).OnlyEnforceIf(b1)
                model.Add(start[e.id] + dur_e > start[f.id]).OnlyEnforceIf(b1.Not())

                # 2) b2: f finishes ≤ e starts
                b2 = model.NewBoolVar(f"f{f.id}_before_e{e.id}")
                model.Add(start[f.id] + dur_f <= start[e.id]).OnlyEnforceIf(b2)
                model.Add(start[f.id] + dur_f > start[e.id]).OnlyEnforceIf(b2.Not())

                # 3) overlap ⇔ not (e before f or f before e)
                overlap = model.NewBoolVar(f"overlap_e{e.id}_f{f.id}_r{r.id}")
                model.AddBoolAnd([b1.Not(), b2.Not()]).OnlyEnforceIf(overlap)
                model.AddBoolOr([b1, b2]).OnlyEnforceIf(overlap.Not())

                # 4) if both in room r AND they overlap, force same start
                model.Add(start[e.id] == start[f.id]).OnlyEnforceIf([
                    in_room[(e.id, r.id)],
                    in_room[(f.id, r.id)],
                    overlap
                ]) """

    # (6) Dept/Year no-overlap
    dept_year_intervals = {}
    for c in Course.course_list:
        for dep in c.departments:
            key = (dep.id, c.year)
            dept_year_intervals.setdefault(key, []).append(interval[c.id])

    for intervals in dept_year_intervals.values():
        model.AddNoOverlap(intervals)
    
    # (6.1) Mandatory exams in the same department cannot overlap
    mandatory_dep_intervals = {}
    for c in Course.course_list:
        if not c.mandatory:
            continue
        for dep in c.departments:
            key = dep.id
            mandatory_dep_intervals.setdefault(key, []).append(interval[c.id])

    for intervals in mandatory_dep_intervals.values():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # (7) Lab vs non-lab enforcement
    for e in Course.course_list:
        if e.requires_lab:
            for r in Room.room_list:
                if not r.is_lab:
                    model.Add(seat[(e.id, r.id)] == 0)
        else:
            for r in Room.room_list:
                if r.is_lab:
                    model.Add(seat[(e.id, r.id)] == 0)

    # (8) Link "in_room ⇒ seat ≥ 1"
    for e in Course.course_list:
        for r in Room.room_list:
            model.Add(seat[(e.id, r.id)] >= 1).OnlyEnforceIf(in_room[(e.id, r.id)])

    # (9) Day-within-slot constraints
    for e in Course.course_list:
        day_e = model.NewIntVar(0, num_days - 1, f"day_e{e.id}")
        model.AddDivisionEquality(day_e, start[e.id], slots_per_day)
        model.Add(start[e.id] >= day_e * slots_per_day)
        model.Add(start[e.id] + e.get_duration() <= (day_e + 1) * slots_per_day)

    # (10) Off-times
    off_chunks = get_off_chunks(TimeSlot.slot_list)
    for e in Course.course_list:
        day_e = model.NewIntVar(0, num_days - 1, f"day_off_e{e.id}")
        model.AddDivisionEquality(day_e, start[e.id], slots_per_day)

        for (o_start, o_end) in off_chunks:
            b1 = model.NewBoolVar(f"b1_e{e.id}_{o_start}_{o_end}")
            b2 = model.NewBoolVar(f"b2_e{e.id}_{o_start}_{o_end}")
            model.Add(end[e.id] <= o_start).OnlyEnforceIf(b1)
            model.Add(start[e.id] >= o_end).OnlyEnforceIf(b2)
            model.AddBoolOr([b1, b2])

    # (11) Balanced exam-per-department constraints
    local_day = {}
    if is_midterm:
        week_len = num_days // 2
        for e in Course.course_list:
            local_day[e.id] = model.NewIntVar(0, week_len - 1, f"local_day_e{e.id}")
            if e.year in {1, 3}:
                model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)
            else:
                dur = e.get_duration()
                shifted_start = model.NewIntVar(0, horizon - slots_per_day * week_len - dur, f"shifted_start_{e.id}")
                model.Add(shifted_start == start[e.id] - slots_per_day * week_len)
                model.AddDivisionEquality(local_day[e.id], shifted_start, slots_per_day)
    else:
        week_len = num_days
        for e in Course.course_list:
            local_day[e.id] = model.NewIntVar(0, week_len - 1, f"local_day_e{e.id}")
            model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)

    count_vars = {}
    for dep in Department.departments:
        for year, exams in dep.curriculums.items():
            for d in range(week_len):
                count_vars[(dep.id, year, d)] = model.NewIntVar(0, len(exams), f"count_dep{dep.id}_yr{year}_d{d}")
                indicators = []
                for e in exams:
                    ind = model.NewBoolVar(f"ind_e{e.id}_d{d}")
                    model.Add(local_day[e.id] == d).OnlyEnforceIf(ind)
                    model.Add(local_day[e.id] != d).OnlyEnforceIf(ind.Not())
                    indicators.append(ind)
                model.Add(count_vars[(dep.id, year, d)] == sum(indicators))

    tolerance = 1
    for dep in Department.departments:
        for year, exams in dep.curriculums.items():
            avg = len(exams) / week_len
            low = int(avg)
            high = int(avg) + tolerance
            for d in range(week_len):
                model.Add(count_vars[(dep.id, year, d)] >= low)
                model.Add(count_vars[(dep.id, year, d)] <= high)

    """ # (11) Balanced exam-per-department constraints (SOFT, overload only)
    local_day = {}
    if is_midterm:
        week_len = num_days // 2
        for e in Course.course_list:
            local_day[e.id] = model.NewIntVar(0, week_len - 1, f"local_day_e{e.id}")
            if e.year in {1, 3}:
                model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)
            else:
                dur = e.get_duration()
                shifted_start = model.NewIntVar(
                    0,
                    horizon - slots_per_day * week_len - dur,
                    f"shifted_start_{e.id}"
                )
                model.Add(shifted_start == start[e.id] - slots_per_day * week_len)
                model.AddDivisionEquality(local_day[e.id], shifted_start, slots_per_day)
    else:
        week_len = num_days
        for e in Course.course_list:
            local_day[e.id] = model.NewIntVar(0, week_len - 1, f"local_day_e{e.id}")
            model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)

    count_vars = {}
    for dep in Department.departments:
        for year, exams in dep.curriculums.items():
            for d in range(week_len):
                count_vars[(dep.id, year, d)] = model.NewIntVar(
                    0, len(exams), f"count_dep{dep.id}_yr{year}_d{d}"
                )
                indicators = []
                for e in exams:
                    ind = model.NewBoolVar(f"ind_e{e.id}_d{d}")
                    model.Add(local_day[e.id] == d).OnlyEnforceIf(ind)
                    model.Add(local_day[e.id] != d).OnlyEnforceIf(ind.Not())
                    indicators.append(ind)
                model.Add(count_vars[(dep.id, year, d)] == sum(indicators))

    # Soft penalties: discourage too many exams on the same day
    overload_penalties = []

    for dep in Department.departments:
        for year, exams in dep.curriculums.items():
            n = len(exams)
            if n == 0:
                continue

            base = n // week_len
            cap = base + 1

            for d in range(week_len):
                overload = model.NewIntVar(0, n, f"over_dep{dep.id}_yr{year}_d{d}")
                model.Add(overload >= count_vars[(dep.id, year, d)] - cap)
                model.Add(overload >= 0)
                overload_penalties.append(overload) """

    if not demo_mode:
        # (12) Mission Active vars
        mission_active = {}
        for r in Room.room_list:
            for t in TimeSlot.slot_list:
                var = model.NewBoolVar(f"mission_active_r{r.id}_t{t.id}")
                mission_active[(r.id, t.id)] = var * 2 if r.id in Room.special_ids else var

                if r.off_times and t.id in r.off_times:
                    model.Add(var == 0)

                active_list = []
                for e in Course.course_list:
                    b_start = model.NewBoolVar(f"bstart_e{e.id}_{r.id}_before_{t.id}")
                    b_end = model.NewBoolVar(f"bend_e{e.id}_{r.id}_after_{t.id}")
                    model.Add(start[e.id] <= t.id).OnlyEnforceIf(b_start)
                    model.Add(start[e.id] > t.id).OnlyEnforceIf(b_start.Not())
                    model.Add(end[e.id] > t.id).OnlyEnforceIf(b_end)
                    model.Add(end[e.id] <= t.id).OnlyEnforceIf(b_end.Not())

                    active = model.NewBoolVar(f"active_e{e.id}_r{r.id}_t{t.id}")
                    model.AddBoolAnd([in_room[(e.id, r.id)], b_start, b_end]).OnlyEnforceIf(active)
                    model.AddBoolOr([in_room[(e.id, r.id)].Not(), b_start.Not(), b_end.Not()]).OnlyEnforceIf(active.Not())
                    active_list.append(active)
                model.AddMaxEquality(var, active_list)

        total_missions = sum(mission_active.values())
        room_usage = [in_room[(e.id, r.id)] for e in Course.course_list for r in Room.room_list]

        balance_weight = 2
        #model.Minimize(sum(room_usage) + 2 * total_missions + balance_weight * sum(overload_penalties))
        model.Minimize(sum(room_usage) + 2 * total_missions)
    else:
        room_usage = [in_room[(e.id, r.id)] for e in Course.course_list for r in Room.room_list]
        model.Minimize(sum(room_usage))

    exam_type_str = "midterm" if is_midterm else "final"
    demo_filename = f"demo_{exam_type_str}_{dataset_name}.json"

    if demo_mode:
        print("Loading variables...")
        success = load_demo_hints(demo_filename, model, start, in_room)

        if not success:
            print("Fallback: Running normal optimization (File missing).")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30 if demo_mode and success else timeout
    solver.parameters.num_search_workers = 12
    solver.parameters.log_search_progress = True
    solver.parameters.symmetry_level = 3

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Solution status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}")

        if not demo_mode:
            save_demo_data(demo_filename, start, in_room, solver)

        exam_type = "_midterm" if is_midterm else "_final"
        exp_path = os.path.join(runs_path, "exp" + str(experiment_no) + exam_type)
        os.makedirs(exp_path, exist_ok=True)

        build_timetable(Course.course_list, Room.room_list, horizon, num_days, solver, start, in_room, seat, exp_path)
        department_exam_schedule(Department.departments, Course.course_list, Room.room_list, horizon, num_days, solver, start, in_room, exp_path)
        faculty_exam_schedule(Course.course_list, Room.room_list, horizon, num_days, solver, start, in_room, exp_path)

        # mission_report(solver, start, slots_per_day, in_room, num_days)
        excelify(Department.departments, exp_path, Room.room_list, output_filename="exam_schedule.xlsx")
    else:
        print(f"No solution found (status {status}).")
