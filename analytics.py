import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from entities import Course, Room
from utils import read_fac_xlsxs, automated_df_rebuild

def plot_exam_per_day(exams_per_day, num_days):
    x = list(range(1, num_days + 1))
    y = [exams_per_day.get(day - 1, 0) for day in x]
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", label="Total Exams")
    plt.xlabel("Day")
    plt.ylabel("Number of Exams")
    plt.title("Exams Scheduled per Day")
    plt.legend()
    plt.xticks(x)
    plt.show()

def plot_dep_year_exam_counts(dep_year_per_day, num_days):
    keys = dep_year_per_day.keys()
    dep_set = sorted({dep_short for (_, dep_short, _) in keys})
    fig, axes = plt.subplots(len(dep_set), 1, figsize=(12, 3 * len(dep_set)), sharex=True)
    years = [1, 2, 3, 4]
    year_colors = ["blue", "orange", "green", "purple"]

    for idx, dep_short in enumerate(dep_set):
        ax = axes[idx]
        for year in years:
            y_vals = [dep_year_per_day.get((day, dep_short, year), 0) for day in range(num_days)]
            nonzero = [(day, count) for day, count in zip(range(num_days), y_vals) if count != 0]
            if nonzero:
                days_f, vals_f = zip(*nonzero)
                ax.plot(days_f, vals_f, marker="o", color=year_colors[year - 1], label=f"Year {year}")
        ax.set_title(dep_short)
        ax.set_ylabel("Exam Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    plt.xlabel("Day")
    plt.xticks(range(num_days), [f"Day {d}" for d in range(num_days)])
    plt.suptitle("Exam Distribution by Department and Year per Day", y=1.02)
    plt.tight_layout()
    plt.show()

def mission_report(solver, start_vars, slots_per_day, in_room_vars, num_days):
    exams_per_day = {}
    for e in Course.course_list:
        s_val = solver.Value(start_vars[e.id])
        day_idx = s_val // slots_per_day
        exams_per_day[day_idx] = exams_per_day.get(day_idx, 0) + 1

    print("Exams scheduled per day:")
    total_exams = 0
    for day, count in sorted(exams_per_day.items()):
        print(f"  Day {day}: {count} exams")
        total_exams += count
    plot_exam_per_day(exams_per_day, num_days)
    print(f"Total exams: {total_exams}\n")

    dep_year_per_day = {}
    for e in Course.course_list:
        s_val = solver.Value(start_vars[e.id])
        day_idx = s_val // slots_per_day
        key = (day_idx, e.departments[0].short, e.year)
        dep_year_per_day[key] = dep_year_per_day.get(key, 0) + 1
    plot_dep_year_exam_counts(dep_year_per_day, num_days)

def find_unique_rooms(df: pd.DataFrame) -> list:
    all_rooms_list = df['Assigned Rooms'].str.split(',').explode().to_list()
    unique_rooms = {Room.find_by_code(code) for code in all_rooms_list}
    return list(unique_rooms)

def room_utilization(df: pd.DataFrame, is_manuel: bool = False):
    """
    Calculates room utilization metrics by day and overall.
    
    This version uses pandas' groupby for more efficient and readable
    processing, eliminating the need for nested explicit loops.
    """
    def calculate_total_capacity(df):
        rooms = find_unique_rooms(df)
        return sum(room.capacity if room.is_lab else room.capacity // 2 for room in rooms)
    
    def count_unique_rooms(df):
        return len(df['Assigned Rooms'].str.split(',').explode().unique())
        
    def find_flawed_entries_by_day(df):
        flawed_entries = df[df['Num Students'] > df['Total Room Cap']]
        return flawed_entries.groupby('Day').size()

    def calculate_excess_students_by_day(df):
        # Filter for entries where student count exceeds capacity
        excess_entries = df[df['Num Students'] > df['Total Room Cap']]
        # Calculate the excess students for each flawed entry
        excess_students = excess_entries['Num Students'] - excess_entries['Total Room Cap']
        # Group the excess students by day and sum them
        return excess_students.groupby(excess_entries['Day']).sum()

    if not is_manuel:
        slot_groups = df.groupby(['Day', 'Starting Slot'])
        slot_capacities = slot_groups.apply(calculate_total_capacity)
        slot_student_numbers = slot_groups['Num Students'].sum()
        slot_unique_rooms = slot_groups.apply(count_unique_rooms)
        daily_capacities = slot_capacities.groupby('Day').sum()
        daily_student_numbers = slot_student_numbers.groupby('Day').sum()
        unique_rooms_per_day = df.groupby('Day').apply(count_unique_rooms)
        daily_active_room_time_slots = slot_unique_rooms.groupby('Day').sum()
    else:
        daily_capacities = df.groupby('Day')['Total Room Cap'].sum()
        daily_student_numbers = df.groupby('Day')['Num Students'].sum()
        unique_rooms_per_day = df.groupby('Day')['Num Rooms Used'].sum()
        daily_active_room_time_slots = None
        
        flawed_entries_per_day = find_flawed_entries_by_day(df)
        print(f"Total number of infeasbile assignments: {sum(dict(flawed_entries_per_day).values())}")
        
        # Calculate and print the daily sum of excess students
        daily_excess_students = calculate_excess_students_by_day(df)
        print(f"Daily Sum of Excess Students: {dict(daily_excess_students)}")
        daily_student_numbers = daily_student_numbers - daily_excess_students * 2

    daily_utilizations = daily_student_numbers / daily_capacities
    overall_utilization = daily_student_numbers.sum() / daily_capacities.sum()

    print(f"Daily Capacities: {list(daily_capacities)}")
    print(f"Daily Student Numbers: {list(daily_student_numbers)}")
    print(f"Overall Utilization: {overall_utilization}")
    print(f"Daily Utilizations: {list(daily_utilizations)}")
    
    if is_manuel:
        active_slots = sum(unique_rooms_per_day)
        print(f"Daily Active Room-Time Slots: {list(unique_rooms_per_day)}")
        print(f"Total Active Room-Time Slots: {sum(unique_rooms_per_day)}")
    else:
        active_slots = sum(daily_active_room_time_slots)
        print(f"Daily Active Room-Time Slots: {list(daily_active_room_time_slots)}")
        print(f"Total Active Room-Time Slots: {active_slots}")
        
    return active_slots, overall_utilization
def analyze_exam_distribution(df: pd.DataFrame, is_midterm: bool):
    df_processed = df.assign(Departments=df['Departments'].str.split(',')).explode('Departments')
    grouped = df_processed.groupby(['Departments', 'Year'])
    distribution_std_dev = {}

    for (department, year), group in grouped:
        daily_counts = group.groupby('Day').size()
        if is_midterm:
            relevant_days = range(1, 6) if year in [1, 3] else range(6, 11)
        else:
            num_days = int(df['Day'].max())
            relevant_days = range(1, num_days + 1)
        
        daily_counts = daily_counts.reindex(list(relevant_days), fill_value=0)
        distribution_std_dev[(department, year)] = np.std(daily_counts.values)

    return distribution_std_dev

def get_daily_exam_counts(df: pd.DataFrame, is_midterm: bool):
    df_processed = df.assign(Departments=df['Departments'].str.split(',')).explode('Departments')
    grouped = df_processed.groupby(['Departments', 'Year'])
    daily_counts_dict = {}
    num_days = int(df['Day'].max())
    full_index = list(range(1, num_days + 1))

    for (department, year), group in grouped:
        base_counts = group.groupby('Day').size()
        if is_midterm:
            relevant_days = range(1, 6) if year in [1, 3] else range(6, 11)
        else:
            relevant_days = full_index

        full_series = pd.Series(0, index=full_index, dtype=int)
        full_series.loc[list(relevant_days)] = base_counts.reindex(list(relevant_days), fill_value=0)
        daily_counts_dict[(department, year)] = full_series.to_list()

    return daily_counts_dict

def frequency_table(experiment_no: int, exam: str, num_days):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import ast

    # ===============================
    # 1) LOAD BEAUTIFIED TIMETABLE
    # ===============================
    exp_path = f"./runs/exp{experiment_no}_{exam[:-1]}"
    timetable_xlsx = "beautified_exam_schedule.xlsx"
    timetable_path = os.path.join(exp_path, timetable_xlsx)
    timetable_df = pd.read_excel(timetable_path, index_col=None, header=0)

    # ===============================
    #  AUTOMATED TIMETABLE REBUILD
    # ===============================
    automated_xlsx = "faculty_schedule.xlsx"
    automated_path = os.path.join(exp_path, automated_xlsx)
    automated_df = pd.read_excel(automated_path, index_col=None, header=0)

    final_automated_df = automated_df_rebuild(automated_df, num_days)
    final_automated_df.to_excel("faculty_schedule_automated_finals.xlsx")

    # --- Sort automated timetable like manual ---
    automated_clean = final_automated_df.replace("", pd.NA)
    usage_automated = automated_clean.notna().sum(axis=0).sort_values(ascending=False)

    auto_sorted_cols = usage_automated.index
    auto_sorted_df = final_automated_df[auto_sorted_cols]

    auto_final_df = auto_sorted_df.T
    auto_final_df = auto_final_df.loc[
        auto_final_df.replace("", pd.NA).notna().sum(axis=1).sort_values(ascending=False).index
]

    auto_final_df.to_excel("faculty_schedule_automated_finals_sorted.xlsx")

    # Compute usage for beautified schedule
    cols = timetable_df.columns.drop("Day")
    usage_beautified = (
        timetable_df[cols]
        .notna()
        .sum()
        .sort_values(ascending=False)
    )

    # ========================================================
    # 2) REBUILD TIMETABLE FROM MANUAL FACULTY SCHEDULE
    # ========================================================
    path_in = "./data/department_schedules_finals/faculty_schedule_manuel_finals.xlsx"
    path_out = "./faculty_schedule_manuel_finals_timetable.xlsx"

    df = pd.read_excel(path_in)

    # Parse Assigned Rooms string → Python list
    df["RoomsList"] = df["Assigned Rooms"].apply(ast.literal_eval)

    # Collect all unique rooms
    rooms = sorted({r for lst in df["RoomsList"] for r in lst})

    # Create timetable (64 rows, one per timeslot)
    n_rows = 8 * 8  # 8 days * 8 slots
    timetable = pd.DataFrame("", index=range(1, n_rows + 1), columns=rooms)

    # Track next free timeslot for each room
    room_next_row = {room: 1 for room in rooms}

    # Fill timetable
    for _, row in df.sort_values("Day").iterrows():
        course_id = row["Course ID"]
        for room in row["RoomsList"]:
            start = room_next_row[room]
            timetable.loc[start, room] = course_id
            timetable.loc[start + 1, room] = course_id
            room_next_row[room] += 2

    # ========================================================
    # 3) Collapse every 2 timeslots → 1 exam block
    # ========================================================
    merged = pd.DataFrame(columns=timetable.columns)

    for i in range(0, len(timetable), 2):
        block = {}
        for room in timetable.columns:
            top = timetable.iloc[i][room]
            bottom = timetable.iloc[i + 1][room]
            block[room] = top if top != "" else bottom
        merged.loc[i // 2] = block

    merged.index = range(1, len(merged) + 1)

    # ========================================================
    # 4) Sort columns (optional), transpose, sort rows by usage
    # ========================================================
    # Sort columns by usage before transpose
    sorted_cols = merged.notna().sum().sort_values(ascending=False).index
    merged_sorted = merged[sorted_cols]

    # Transpose -> rows are rooms
    final_df = merged_sorted.T

    # Treat empty strings as NaN
    clean = final_df.replace("", pd.NA)

    # Compute usage per room
    usage_generated = clean.notna().sum(axis=1)

    # Sort rooms by usage
    sorted_rows = usage_generated.sort_values(ascending=False).index
    final_df = final_df.loc[sorted_rows]

    # Save timetable
    final_df.to_excel(path_out, index_label="Room")

    # ========================================================
    # 5) TWO-WAY USAGE COMPARISON PLOT (Manual vs Automated)
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    max_y = max(
        usage_generated.max(),
        usage_automated.max()
    )

    # ------------------------------
    # Plot 1: Manual Schedule
    # ------------------------------
    axes[0].bar(
        usage_generated.sort_values(ascending=False).index,
        usage_generated.sort_values(ascending=False).values
    )
    axes[0].set_title("Manual Schedule — Room Usage")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set_ylim(0, max_y)
    axes[0].set_xlabel("Rooms")
    axes[0].set_ylabel("Mission Count")

    # ------------------------------
    # Plot 2: Automated Schedule
    # ------------------------------
    axes[1].bar(usage_automated.index, usage_automated.values)
    axes[1].set_title("Automated Schedule — Room Usage")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_ylim(0, max_y)
    axes[1].set_xlabel("Rooms")
    axes[1].set_ylabel("Mission Count")

    plt.tight_layout()
    plt.show()


    import seaborn as sns
    import numpy as np


    # Convert strings → 1 (occupied) and empty → 0
    def to_binary(df):
        return df.replace("", np.nan).notna().astype(int)

    h_beautified = to_binary(timetable_df.drop(columns=["Day"]))
    h_manual     = to_binary(final_df)
    h_auto       = to_binary(auto_final_df)

    # --- Build mapping: room_code → "room_code (capacity)" ---
    room_label_map = {
        room.room_code: f"{room.room_code} ({room.capacity // 2 if not room.is_lab else room.capacity})"
        for room in Room.room_list
    }

    # --- Rename rows for manual + automated ---
    h_manual.index = [room_label_map.get(idx, idx) for idx in h_manual.index]
    h_auto.index   = [room_label_map.get(idx, idx) for idx in h_auto.index]

    # Plot 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    vmax = 1

    sns.heatmap(h_beautified, ax=axes[0], cmap="Blues", cbar=False, vmax=vmax)
    axes[0].set_title("Heatmap — My Method (Beautified)")
    axes[0].set_xlabel("Rooms")
    axes[0].set_ylabel("Timeslots / Blocks")

    sns.heatmap(h_manual, ax=axes[1], cmap="Greens", cbar=False, vmax=vmax)
    axes[1].set_title("Heatmap — Manual Schedule")
    axes[1].set_xlabel("Blocks")
    axes[1].set_ylabel("Rooms")

    sns.heatmap(h_auto, ax=axes[2], cmap="Reds", cbar=False, vmax=vmax)
    axes[2].set_title("Heatmap — Automated Schedule")
    axes[2].set_xlabel("Blocks")
    axes[2].set_ylabel("Rooms")

    # -----------------------------
    # KEEP SAME X-AXIS SCALE FOR MANUAL & AUTOMATED
    # -----------------------------

    # both must use the same number of columns
    max_x = max(h_manual.shape[1], h_auto.shape[1])

    # force same x-axis range and ticks
    for ax in [axes[1], axes[2]]:
        ax.set_xlim(0, max_x)
        ax.set_xticks(range(max_x))

    plt.tight_layout()
    plt.show()
    print("Timetable saved to:", path_out)

def merge_dfs(fac_df_manuel: pd.DataFrame, fac_df_out: pd.DataFrame, path) -> pd.DataFrame:
    sort_keys = ['Departments', 'Course ID']
    fac_df_manuel_sorted = fac_df_manuel.sort_values(sort_keys).reset_index(drop=True)
    fac_df_out_sorted = fac_df_out.sort_values(sort_keys).reset_index(drop=True)
    
    cols_to_append = ['Day', 'Slot', 'Assigned Rooms', 'Num Rooms Used', 'Total Room Cap', 'Num Students']
    man_cols = fac_df_manuel_sorted[cols_to_append].rename(columns={c: f"MAN_{c}" for c in cols_to_append})
    merged = pd.concat([fac_df_out_sorted, man_cols], axis=1)
    merged.to_excel(path, index=False)
    return merged

def analysis(experiment_no: int, is_midterm: bool, num_days: int):
    exam = "midterms" if is_midterm else "finals"
    fac_df_manuel, fac_df_out = read_fac_xlsxs(experiment_no, exam)
    exp_path = f"./runs/exp{experiment_no}_{exam[:-1]}"
    
    faculty_xlsx = "faculty_schedule2.xlsx"
    faculty_path = os.path.join(exp_path, faculty_xlsx)
    merged = merge_dfs(fac_df_manuel, fac_df_out, faculty_path)

    #frequency_table(experiment_no, exam, num_days)

    print(f"Manual Schedule Entries: {len(fac_df_manuel)}, Automated Schedule Entries: {len(fac_df_out)}")
    print("-" * 30)

    print("--- Room Utilization Analysis (Automated) ---")
    act_ts_au, util_au = room_utilization(fac_df_out)
    print("\n--- Room Utilization Analysis (Manual) ---")
    act_ts_ma, util_ma = room_utilization(fac_df_manuel, is_manuel=True)
    print("-" * 30)

    room_time_improvement = 100 - (act_ts_au / act_ts_ma) * 100
    room_utilization_improvement = 100 - (util_ma / util_au) * 100

    print(f"\nRoom-Time Improvement: {room_time_improvement:.2f}%")
    print(f"Room-Utilization Improvement: {room_utilization_improvement:.2f}%")
    print("-" * 30)

    print("\n--- Exam Distribution Analysis ---")
    dist_manual = analyze_exam_distribution(fac_df_manuel, is_midterm)
    dist_auto = analyze_exam_distribution(fac_df_out, is_midterm)

    avg_std_manual = np.mean(list(dist_manual.values()))
    avg_std_auto = np.mean(list(dist_auto.values()))
    
    distribution_improvement = 100 - (avg_std_auto / avg_std_manual) * 100

    print(f"Average Standard Deviation of Daily Exams (Manual): {avg_std_manual:.2f}")
    print(f"Average Standard Deviation of Daily Exams (Automated): {avg_std_auto:.2f}")
    print(f"Improvement in Exam Distribution Uniformity: {distribution_improvement:.2f}%")

    """ print("\n" + "="*40)
    print("--- Daily Exam Distribution Analysis ---")
    print("="*40)
    
    manual_counts = get_daily_exam_counts(fac_df_manuel, is_midterm)
    auto_counts = get_daily_exam_counts(fac_df_out, is_midterm)

    # Prepare data for a comparison DataFrame
    comparison_data = []
    all_keys = sorted(manual_counts.keys() | auto_counts.keys()) # Use union of keys

    for dept, year in all_keys:
        manual_dist = manual_counts.get((dept, year), 'N/A')
        auto_dist = auto_counts.get((dept, year), 'N/A')
        
        comparison_data.append({
            "Department": dept,
            "Year": year,
            "Manual": manual_dist,
            "Automated": auto_dist
        })
        
    # Create and display the DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Optional: Set pandas display options to see the full lists
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    # --- add 4 faculty-wide rows (sum across departments) ---

    def _elemwise_sum(series):
        lists = [x for x in series if isinstance(x, (list, tuple))]
        if not lists:
            return []
        maxlen = max(len(l) for l in lists)
        total = [0] * maxlen
        for l in lists:
            for i, v in enumerate(l):
                total[i] += int(v)
        return total

    # Existing per-year faculty rows (might be 5-day if you didn't change the function)
    faculty_by_year = (
        comparison_df
        .groupby('Year', sort=True)
        .agg({'Manual': _elemwise_sum, 'Automated': _elemwise_sum})
        .reset_index()
    )
    faculty_by_year.insert(0, 'Department', 'FACULTY')

    # NEW: faculty-wide 10-day roll-up (ALL years combined)
    faculty_all = pd.DataFrame([{
        'Department': 'FACULTY',
        'Year': 'ALL',
        'Manual': _elemwise_sum(comparison_df['Manual']),
        'Automated': _elemwise_sum(comparison_df['Automated']),
    }])

    # Append both to your table
    comparison_df = pd.concat([comparison_df, faculty_by_year, faculty_all], ignore_index=True)

    print("\n=== FACULTY totals (per-year and ALL 10 days) ===")
    print(faculty_by_year.to_string(index=False))
    print(faculty_all.to_string(index=False))
    # print(comparison_df.to_string(index=False)) """